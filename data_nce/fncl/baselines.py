"""fncl.baselines — 同じ NNN 上の代替 forward-only 学習則 (論文 §6.3, Table 4).

いずれも cov_jac と同じ forward サンプル、同じ分布フリー KDE slope、同じ読み出し
誤差を使い、隠れ層 credit の作り方だけが異なる:

  fa  : feedback alignment (Lillicrap et al. 2016)。cov_jac の再帰 (§4.3) の
        weight mirror W_hat を固定ランダム行列 B に置き換えたもの,
  dfa : direct feedback alignment (Nokland 2016)。出力誤差を固定ランダム行列
        B_l で各隠れ層へ直接投影,
  node perturbation: 全ユニットに外部摂動を注入する common-random-number 推定
        (fncl.train の cov_deriv_gate_crn を gate_block_size=H で使う),
  pepita: 誤差変調した 2 回目の forward とアンサンブル平均活性の差で隠れ層を更新
        (Dellaferrera and Kreiman 2022)。

- train_fa / train_pepita: sin(x) と §5.6 ベンチマーク (スカラー出力) 用の学習ループ,
- random_feedback / grads_from: MNIST (多クラス) で同一サンプル上の更新方向を
  cov_jac / FA / DFA で作り比べる (Table 4 最右列の cosine)。
"""
from __future__ import annotations

import math

import torch

from .mnist import ce_error
from .network import Capture, crossing_layers, kde_slope
from .train import ManualOpt


# ============================================================
# FA / DFA (sin(x) とベンチマーク)
# ============================================================
def train_fa(net, x, t_target, method: str, lr: float, epochs: int,
             opt: str = "adam", log_every: int = 0):
    """feedback alignment ("fa") / direct feedback alignment ("dfa") の学習ループ.

    隠れ層 credit:
      fa : a^{L-1} = e B_out,  a^l = (a^{l+1} * slope^{l+1}) B^{l+1}
           (cov_jac の再帰と同形で、W_hat を固定ランダム B に置き換える)
      dfa: a^l = e B_l  (出力誤差を各層へ直接投影)
    読み出し層はアンサンブル平均特徴上の局所勾配 (全手法共通)。
    """
    cap = Capture(net)
    n_hidden = cap.n_hidden
    N = x.shape[0]
    out_dim = net.fcs[-1].weight.shape[0]
    hidden_sizes = list(net.structure[1:-1])

    # 固定ランダムフィードバック行列 (前向き重みの既定初期化と同じ一様スケール)
    B = {}
    if method == "fa":
        for l in range(1, n_hidden):
            fan_in = hidden_sizes[l - 1]
            B[l] = (torch.rand_like(net.fcs[l].weight) * 2 - 1) / math.sqrt(fan_in)
        B["out"] = ((torch.rand_like(net.fcs[-1].weight) * 2 - 1)
                    / math.sqrt(hidden_sizes[-1]))
    else:  # dfa
        for l in range(n_hidden):
            B[l] = ((torch.rand(out_dim, hidden_sizes[l], device=x.device) * 2 - 1)
                    / math.sqrt(hidden_sizes[l]))

    optim = ManualOpt(opt)
    losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            y = net(x)                                       # [N, C]; fires hooks
            d = [cap.d[l] for l in range(n_hidden)]
            z = [cap.z[l] for l in range(n_hidden)]
            T = z[0].shape[1]
            e = 2.0 * (y - t_target)                         # [N, C]
            slope_full = [kde_slope(cap.crossings[l], d[l]) for l in range(n_hidden)]
            slope_mean = [s.mean(dim=1) for s in slope_full]

            a = [None] * n_hidden
            if method == "fa":
                a[-1] = e @ B["out"]
                for l in range(n_hidden - 2, -1, -1):
                    a[l] = (a[l + 1] * slope_mean[l + 1]) @ B[l + 1]
            else:
                for l in range(n_hidden):
                    a[l] = e @ B[l]

            z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
                      + [z[l] for l in range(n_hidden - 1)])
            for l in range(n_hidden):
                delta = a[l].unsqueeze(1) * slope_full[l]     # [N, T, H]
                gW = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
                gb = delta.mean(dim=(0, 1))
                optim.update(f"w{l}", net.fcs[l].weight, gW, lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb, lr)

            z_bar = z[-1].mean(dim=1)
            gWout = torch.einsum("no,ni->oi", e, z_bar) / N
            optim.update("wout", net.fcs[-1].weight, gWout, lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, e.mean(dim=0), lr)

            losses.append(float(((y - t_target) ** 2).mean()))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [{method}/{opt}] epoch {epoch:5d}  mse={losses[-1]:.5f}", flush=True)
    cap.remove()
    return losses


# ============================================================
# PEPITA (誤差変調した 2 回目の forward)
# ============================================================
def train_pepita(net, x, t_target, lr: float, epochs: int, fscale: float,
                 opt: str = "sgd", log_every: int = 0):
    cap = Capture(net)
    n_hidden = cap.n_hidden
    N, in_dim = x.shape
    out_dim = net.fcs[-1].weight.shape[0]
    F = fscale * torch.randn(in_dim, out_dim, device=x.device) / math.sqrt(out_dim)
    optim = ManualOpt(opt)
    losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            y1 = net(x)                                       # 通常の forward
            zbar1 = [cap.z[l].mean(dim=1) for l in range(n_hidden)]
            e = y1 - t_target                                 # [N, C]
            x_mod = x + e @ F.T                               # 誤差変調した入力
            y2 = net(x_mod)                                   # 変調した forward
            zbar2 = [cap.z[l].mean(dim=1) for l in range(n_hidden)]
            prev2 = [x_mod] + zbar2[:-1]
            for l in range(n_hidden):
                g = zbar1[l] - zbar2[l]                       # [N, H]
                gW = torch.einsum("no,ni->oi", g, prev2[l]) / N
                optim.update(f"w{l}", net.fcs[l].weight, gW, lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, g.mean(dim=0), lr)
            gWout = torch.einsum("no,ni->oi", e, zbar2[-1]) / N
            optim.update("wout", net.fcs[-1].weight, gWout, lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, e.mean(dim=0), lr)
            losses.append(float(((y1 - t_target) ** 2).mean()))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [pepita f={fscale}] epoch {epoch:5d}  mse={losses[-1]:.5f}", flush=True)
    cap.remove()
    return losses


# ============================================================
# MNIST 上の更新方向 (cov_jac / FA / DFA を同一サンプルで比較)
# ============================================================
def random_feedback(net, depth: int, method: str, width: int, device) -> dict:
    """FA / DFA の固定ランダムフィードバック行列 (前向き重みと同じ一様スケール)."""
    B = {}
    if method == "fa":
        for l in range(1, depth):
            B[l] = ((torch.rand_like(net.fcs[l].weight) * 2 - 1)
                    / math.sqrt(net.fcs[l].weight.shape[1]))
        B["out"] = ((torch.rand_like(net.fcs[-1].weight) * 2 - 1)
                    / math.sqrt(net.fcs[-1].weight.shape[1]))
    else:  # dfa: [10, H_l] を各隠れ層に
        for l in range(depth):
            B[l] = (torch.rand(10, width, device=device) * 2 - 1) / math.sqrt(width)
    return B


def grads_from(net, x, labels, s: dict, method: str, W: dict = None,
               B: dict = None) -> dict:
    """同一サンプル s 上で、隠れ層 credit の作り方だけを変えた更新方向を返す.

    method: "cov_jac" (weight mirror W の再帰), "fa" (固定 B の再帰), "dfa" (直接投影)。
    """
    crossings = crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    N, T = x.shape[0], s["z"][0].shape[1]
    e = ce_error(s["y"], labels)                                   # [N, 10]
    a = [None] * n_hidden
    if method == "dfa":
        for l in range(n_hidden):
            a[l] = e @ B[l]
    else:
        fb = W if method == "cov_jac" else B
        a[-1] = e @ fb["out"]
        for l in range(n_hidden - 2, -1, -1):
            a[l] = (a[l + 1] * slope_mean[l + 1]) @ fb[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    grads = {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]
        grads[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
    grads["wout"] = torch.einsum("no,ni->oi", e, s["z"][-1].mean(dim=1)) / N
    return grads
