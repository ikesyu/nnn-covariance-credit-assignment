"""fncl.noisefield — サンプル軸に構造を持つノイズ場 (論文 §6.4) の共通部.

交差活性 (nnn/activation.py の CrossingSample) は T 方向の隣接サンプル間の XOR で
あり、連続区間でノイズを固定すると XOR が恒等的に 0 になってユニットが死ぬ。
そこで上流層のノイズを周期 P でタイルする:

    eta_(m) = eta_(m mod P)        (P >= 2, P | T)

隣接サンプルは異なるので交差活性は通常どおり動き、位相クラス {m : m = c mod P} の
中では上流の構成が固定される。対象層の揺らぎはそのユニット固有のノイズだけに
由来するので、位相クラス内で中心化して共分散を取れば単変量 weight mirror の
層内相関バイアスは母集団極限で消える (§6.4)。

- PeriodicGaussianNoise / PeriodicUniformCrossing: 周期ノイズ層,
- install_noise_field / restore_noise_field: 隠れ層ごとの周期の設定と復元,
- phase_center / cov_weight_phase: 位相クラス内で中心化した weight mirror,
- train_cov_jac_ce_noisefield: 1 epoch に 1 層ずつ mirror を測るローテーション学習
  (残りの mirror は Kolen-Pollack 追従)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nnn.activation import CrossingSample as _CrossingSample

from .mnist import (apply_updates, collect_samples, evaluate, fidelity_at,
                    manual_grads, measure_mirrors, mirror_pearson)
from .network import crossing_layers
from .train import ManualOpt, cov_weight

EPS = 1e-8


# ============================================================
# 周期ノイズ層
# ============================================================
class PeriodicGaussianNoise(nn.Module):
    """周期 period でタイルしたガウスノイズを加える (period=1 なら通常の i.i.d.).

    nnn.layer.GaussianNoiseLayer と同じインタフェース (forward(x, std=None)) を持ち、
    `crossing_layer.noise_layer` をそのまま差し替えられる。
    """

    def __init__(self, std, period: int = 1, mean: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.std = std
        self.mean = mean
        self.period = int(period)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, std=None) -> torch.Tensor:
        if std is not None:
            self.std = std
        n, t, d = x.shape
        p = self.period
        if p <= 1:
            eps = torch.randn_like(x)
        else:
            if t % p != 0:
                raise ValueError(f"T={t} must be divisible by period={p}")
            base = torch.randn(n, p, d, device=x.device, dtype=x.dtype)
            eps = base.repeat(1, t // p, 1)      # eta_(m) = eta_(m mod P)
        return x + eps * (self.std * self.scale) + self.mean


class PeriodicUniformCrossing(nn.Module):
    """一様ノイズ版の周期ノイズつき交差活性 (nnn.layer.UniformCrossingSampleLayer 互換).

    一様版の交差層は noise_layer を持たず forward 内でノイズを引くので、モジュール
    ごと置き換える。周期化の考え方はガウス版と同じ。
    """

    def __init__(self, radius, center: float = 0.0, h: float = 0.2,
                 period: int = 1, scale: float = 1.0):
        super().__init__()
        self.radius = radius
        self.center = center
        self.h = h
        self.period = int(period)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, radius=None) -> torch.Tensor:
        r = radius if radius is not None else self.radius
        n, t, d = x.shape
        p = self.period
        if p <= 1:
            u = torch.rand_like(x)
        else:
            if t % p != 0:
                raise ValueError(f"T={t} must be divisible by period={p}")
            u = torch.rand(n, p, d, device=x.device, dtype=x.dtype).repeat(1, t // p, 1)
        eps = self.center + (r * self.scale) * (2.0 * u - 1.0)
        return _CrossingSample.apply(x + eps, self.h)


def install_noise_field(net, periods=None):
    """各隠れ層のノイズを周期ノイズ層に差し替える (periods[l] = 層 l の周期).

    元の層は net._nf_orig に退避し、restore_noise_field() で戻せる。
    """
    crossings = crossing_layers(net)
    n_hidden = len(crossings)
    if periods is None:
        periods = [1] * n_hidden
    uniform = not hasattr(crossings[0], "noise_layer")
    if not hasattr(net, "_nf_orig"):
        # ガウス版は noise_layer を、一様版は交差層モジュールそのものを退避する
        net._nf_orig = [c if uniform else c.noise_layer for c in crossings]
        net._nf_uniform = uniform
    for l, c in enumerate(crossings):
        if uniform:
            o = net._nf_orig[l]
            crossings[l] = PeriodicUniformCrossing(o.radius, o.center, o.h, periods[l])
        else:
            c.noise_layer = PeriodicGaussianNoise(net._nf_orig[l].std, periods[l])
    net._nf_periods = list(periods)
    return net


def restore_noise_field(net):
    if hasattr(net, "_nf_orig"):
        crossings = crossing_layers(net)
        for l, orig in enumerate(net._nf_orig):
            if getattr(net, "_nf_uniform", False):
                crossings[l] = orig
            else:
                crossings[l].noise_layer = orig
        net._nf_periods = [1] * len(net._nf_orig)
    return net


# ============================================================
# 位相クラス内の中心化
# ============================================================
def phase_center(a: torch.Tensor, period: int, pass_len: int = 0) -> torch.Tensor:
    """[N, T, D] を位相クラス (m mod period) ごとに中心化して返す.

    複数パスを T 軸に連結している場合、パスごとに独立な周期ノイズが引かれているので
    位相クラスはパス境界を跨いではならない。pass_len に 1 パスの長さを与える。
    """
    n, t, d = a.shape
    if period <= 1:
        return a - a.mean(dim=1, keepdim=True)
    pl = pass_len if pass_len and pass_len < t else t
    if t % pl != 0 or pl % period != 0:
        raise ValueError(f"T={t}, pass_len={pl}, period={period} are incompatible")
    v = a.view(n, t // pl, pl // period, period, d)      # [N, pass, q, c, D]
    return (v - v.mean(dim=2, keepdim=True)).view(n, t, d)


def cov_weight_phase(d_next: torch.Tensor, z_prev: torch.Tensor,
                     period: int, pass_len: int = 0) -> torch.Tensor:
    """位相クラス内で中心化した単変量 weight mirror (fncl.cov_weight の pool=True と同形)."""
    cd = phase_center(d_next, period, pass_len)
    cz = phase_center(z_prev, period, pass_len)
    t = d_next.shape[1]
    cov = torch.einsum("nto,nti->oi", cd, cz) / t
    var = (cz ** 2).mean(dim=1)                              # [N, Hi]
    return cov / (var.sum(dim=0).unsqueeze(0) + EPS)


# ============================================================
# ローテーション学習 (§6.4 の cov_jac + noise field)
# ============================================================
def _mirror_key(t: int, n_hidden: int):
    """回帰子 z[t] に対応する mirror のキー (manual_grads の規約)."""
    return "out" if t == n_hidden - 1 else t + 1


def train_cov_jac_ce_noisefield(net, x, labels, lr: float, epochs: int, jac_ema: float,
                                eval_every: int, grad_draws: int, log=print,
                                test=(None, None), period: int = 16,
                                mirror_every: int = 1):
    """cov_jac の交差エントロピー版に、周期 P の noise field とローテーション測定を組み込む.

    各 epoch で対象層 t を 1 つ選び、層 0..t-1 のノイズを周期 period でタイルし、
    回帰子 z[t] の mirror だけを位相クラス内中心化で測って EMA 更新する。
    他の mirror はその epoch の重み変化量を Kolen-Pollack 追従で加算する。
    mirror_every > 1 なら mirror の測定を mirror_every epoch に 1 度に間引く。
    period == 1 は通常の cov_jac をローテーション測定で走らせた対照になる。
    """
    n_hidden = len(crossing_layers(net))
    optim = ManualOpt("adam")
    W_ema, history = {}, []
    for epoch in range(epochs):
        measure = (epoch % mirror_every == 0)
        target = (epoch // mirror_every) % n_hidden if measure else None
        if period > 1 and measure and target > 0:
            periods = [period] * target + [1] * (n_hidden - target)
        else:
            periods = [1] * n_hidden
        install_noise_field(net, periods=periods)
        with torch.no_grad():
            s = collect_samples(net, x, passes=1)
            T = s["z"][0].shape[1]
            if not W_ema:
                W_ema.update(measure_mirrors(s, n_hidden))   # 初回は全 mirror を通常測定
            elif measure:
                tgt = s["ys"] if target == n_hidden - 1 else s["d"][target + 1]
                if period > 1 and target > 0:
                    meas = cov_weight_phase(tgt, s["z"][target], period, pass_len=T)
                else:
                    meas = cov_weight(tgt, s["z"][target], pool=True)
                k = _mirror_key(target, n_hidden)
                W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * meas
            gW, gb = manual_grads(net, x, labels, W_ema, s)
            steps = apply_updates(net, optim, gW, gb, lr, n_hidden)
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]
        restore_noise_field(net)
        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = evaluate(net, x, labels, *test)
            mirror_r = mirror_pearson(net, W_ema, n_hidden)
            fid = fidelity_at(net, x, labels, W_ema, grad_draws)
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te_acc,
                            "mirror_r": mirror_r, "fidelity": fid})
            log(f"  [noise field P={period} ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} "
                f"test={te_acc:.3f} | mirror "
                + " ".join(f"{k}={v:.3f}" for k, v in mirror_r.items())
                + " | cos " + " ".join(f"{k}={v['cos']:.3f}" for k, v in fid.items()))
    return history
