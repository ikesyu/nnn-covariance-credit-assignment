"""fncl.mirror — weight mirror の推定法を差し替えられる cov_jac 学習ループ (論文 §6.2, §6.3).

train_jac_flex の mirror 引数:
  intrinsic  : 本論文の cov_jac。推論のために既に引いた forward 揺らぎの共分散
               W_hat = sum_n Cov_t(d, z) / sum_n Var_t(z) (追加の forward なし),
  mv         : 同じ揺らぎからの多変量回帰 Cov(d, z) Cov(z, z)^{-1}
               (層内相関のバイアスを除く参照。O(H^3)),
  probe_reg  : 専用フェーズの weight mirror (Akrout et al. 2019 の mirror mode に相当)。
               各層の入力に等方ガウス摂動ベクトルを K 本注入し、その層の前活性
               d = W delta + b を読んで回帰 Cov_K(d, delta)/Var_K(delta),
  probe_hebb : 同じ専用フェーズで Hebb 型の非中心化推定 (d delta^T)/(K alpha^2)。
いずれも EMA 平滑化 (jac_ema) と Kolen-Pollack 追従 (kp_track) は共通。
読み出し誤差は解析形 2(y - t) (二乗損失、スカラー出力)。
probe_* に要する層あたり K 本の線形層評価は extra_evals に数える。
"""
from __future__ import annotations

import torch

from .constants import EPS
from .metrics import pearson
from .network import Capture, kde_slope
from .train import ManualOpt, cov_weight


# ============================================================
# mirror 推定量
# ============================================================
def mv_mirror(d_next: torch.Tensor, z_prev: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """多変量 weight mirror: 入力内で中心化した揺らぎの Cov(d, z) Cov(z, z)^{-1} (全入力でプール).

    ridge は Cov(z, z) の対角平均に対する比率で、数値安定化のために加える。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)              # [N, T, Ho]
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)              # [N, T, Hi]
    NT = cd.shape[0] * cd.shape[1]
    cov_dz = torch.einsum("nto,nti->oi", cd, cz) / NT           # [Ho, Hi]
    cov_zz = torch.einsum("ntk,nti->ki", cz, cz) / NT           # [Hi, Hi]
    eye = torch.eye(cov_zz.shape[0], device=cov_zz.device)
    scale = float(cov_zz.diagonal().mean()) + 1e-12
    return torch.linalg.solve(cov_zz + ridge * scale * eye, cov_dz.T).T


def probe_mirror(linear, K: int, alpha: float, hebb: bool) -> torch.Tensor:
    """専用フェーズの weight mirror。linear = net.fcs[l] (d = W delta + b を読む)."""
    W = linear.weight
    delta = alpha * torch.randn(K, W.shape[1], device=W.device, dtype=W.dtype)
    d = linear(delta)                                           # [K, Ho]
    if hebb:
        return (d.T @ delta) / (K * alpha * alpha)              # 非中心化 Hebb 推定
    cd = d - d.mean(dim=0, keepdim=True)
    cdl = delta - delta.mean(dim=0, keepdim=True)
    return (cd.T @ cdl) / ((cdl ** 2).sum(dim=0).unsqueeze(0) + EPS)


# ============================================================
# 学習ループ (cov_jac の再帰。読み出し誤差は解析形 2(y - t))
# ============================================================
def train_jac_flex(net, x: torch.Tensor, t: torch.Tensor, lr: float, epochs: int,
                   opt: str = "adam", mirror: str = "intrinsic", jac_ema: float = 0.9,
                   kp_track: bool = True, probe_k: int = 16, probe_alpha: float = 0.1,
                   mv_ridge: float = 1e-3, eval_every: int = 50, log_every: int = 0,
                   tag: str = ""):
    """返り値: (losses, history[{epoch, mse, mirror_r{...}}], extra_evals).

    mirror="intrinsic", kp_track=True は fncl.train_cov(method="cov_jac", jac_track=True)
    と同じ更新列を与える。
    """
    assert mirror in ("intrinsic", "mv", "probe_reg", "probe_hebb")
    cap = Capture(net)
    n_hidden = cap.n_hidden
    N = x.shape[0]
    optim = ManualOpt(opt)
    W_ema, losses, history = {}, [], []
    extra_evals = 0
    mirror_keys = ["out"] + list(range(1, n_hidden))

    def true_w(k):
        return net.fcs[-1].weight if k == "out" else net.fcs[k].weight

    for epoch in range(epochs):
        with torch.no_grad():
            y = net(x)                                           # [N, 1]; fires hooks
            ys = cap.y_samples                                   # [N, T, 1]
            z = [cap.z[l] for l in range(n_hidden)]
            d = [cap.d[l] for l in range(n_hidden)]
            T = z[0].shape[1]
            slope_full = [kde_slope(cap.crossings[l], d[l]) for l in range(n_hidden)]
            slope_mean = [s.mean(dim=1) for s in slope_full]

            # --- mirror の測定 ---
            meas = {}
            for k in mirror_keys:
                d_next = ys if k == "out" else d[k]
                z_prev = z[-1] if k == "out" else z[k - 1]
                if mirror == "intrinsic":
                    meas[k] = cov_weight(d_next, z_prev, pool=True)
                elif mirror == "mv":
                    meas[k] = mv_mirror(d_next, z_prev, mv_ridge)
                else:
                    linear = net.fcs[-1] if k == "out" else net.fcs[k]
                    meas[k] = probe_mirror(linear, probe_k, probe_alpha,
                                           hebb=(mirror == "probe_hebb"))
                    extra_evals += probe_k
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * v

            # --- 再帰 credit ---
            err_out = 2.0 * (y - t)                              # [N, 1]
            a = [None] * n_hidden
            a[-1] = err_out * W_ema["out"]                       # [N, H]
            for l in range(n_hidden - 2, -1, -1):
                a[l] = (a[l + 1] * slope_mean[l + 1]) @ W_ema[l + 1]
            z_prev_all = [x.unsqueeze(1).expand(N, T, x.shape[1])] + z[:-1]
            steps = {}
            for l in range(n_hidden):
                delta = a[l].unsqueeze(1) * slope_full[l]        # [N, T, H]
                gW = torch.einsum("nto,nti->oi", delta, z_prev_all[l]) / (N * T)
                gb = delta.mean(dim=(0, 1))
                steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW, lr)
                if net.fcs[l].bias is not None:
                    optim.update(f"b{l}", net.fcs[l].bias, gb, lr)
            z_bar = z[-1].mean(dim=1)
            gWout = torch.einsum("no,ni->oi", err_out, z_bar) / N
            steps["out"] = optim.update("wout", net.fcs[-1].weight, gWout, lr)
            if net.fcs[-1].bias is not None:
                optim.update("bout", net.fcs[-1].bias, err_out.mean(dim=0), lr)
            if kp_track:
                for k in mirror_keys:
                    W_ema[k] = W_ema[k] - steps[k]
            losses.append(float(((y - t) ** 2).mean()))

        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            mr = {str(k): pearson(W_ema[k], true_w(k)) for k in mirror_keys}
            history.append({"epoch": epoch, "mse": losses[-1], "mirror_r": mr})
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [{tag or mirror}] epoch {epoch:5d}  mse={losses[-1]:.5f}", flush=True)
    cap.remove()
    return losses, history, extra_evals
