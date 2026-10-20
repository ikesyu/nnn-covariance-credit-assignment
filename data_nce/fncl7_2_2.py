"""
fncl7_2_2.py — 論文 §7.2 の離散 (b ビット) 一様ノイズ実験 (Fig. 7)

デジタル乱数生成器の出力は連続一様分布ではなく 2^b 水準の離散分布である。§3.1 の F は
階段関数になり、期待応答 phi_bar = 2F(1-F) も階段状になる (Fig. 7(a))。
Table 7 の一様ノイズ設定 (半幅 r=1.0, 他は §5.1 プロトコル) で、ノイズを

  iid-b  : 独立な b ビット量子化一様サンプル (量子化の効果だけを見る対照)
  lfsr-b : 16 ビット最大長 LFSR の上位 b ビット。サンプルごとに b 回クロックする
           (系列相関まで含むデジタル生成器そのもの; Fig. 7(b))

に置き換え、b = 1, 2, 3, 4, 6, 8, 16 と連続 (cont) で backprop / cov_deriv / cov_jac /
cov_jac_full の最終 MSE (3 seed) と、先頭 seed の未学習状態の勾配 cosine・weight mirror
の r を比較する。KDE slope は分布を参照しないので学習則の変更はない
(nnn.layer.UniformCrossingSampleLayer の bits / lfsr と nnn.noise.LFSRUniform を使う)。

生成物 (out/fncl7_2_2/):
  results.json                -> 数値一式 (fncl7_2_2_fig.py が Fig. 7 を描く)
  fig_discrete_response.png   -> 補助図 (離散 F と phi_bar の階段形状)
  (標準出力)                  -> 条件ごとの数表

実行例:
  python data_nce/fncl7_2_2.py --device cuda
  python data_nce/fncl7_2_2.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from fncl_common import (add_common_args, config_dict, fidelity_untrained,  # noqa: E402
                         finalize_args, make_task, mean_std, model_factory, run_method,
                         save_json, savefig)
from nnn.noise import LFSRUniform  # noqa: E402

METHODS = [
    ("backprop",          {"kind": "backprop"}),
    ("cov_deriv_kde",     {"method": "cov_deriv", "slope": "kde"}),
    ("cov_jac_adam",      {"method": "cov_jac", "opt": "adam", "jac_track": True}),
    ("cov_jac_full_adam", {"method": "cov_jac_full", "opt": "adam",
                           "jac_track": True, "jac_out": "cov_m3"}),
]
BITS = [1, 2, 3, 4, 6, 8, 16]


def configure(net, gen: str, bits: int, seed: int):
    """新しいネットの一様交差層に離散ノイズ設定を施す (gen: cont / iid / lfsr)."""
    for i, cl in enumerate(net.uniform_crossing):
        cl.bits = 0
        cl.lfsr = None
        if gen == "iid":
            cl.bits = bits
        elif gen == "lfsr":
            H = net.structure[i + 1]
            cl.lfsr = LFSRUniform(H, bits, next(net.parameters()).device,
                                  seed=1000 * seed + i)
    return net


def response_figure(radius: float, path: Path):
    """離散一様ノイズの F (階段) と phi_bar = 2F(1-F) を連続の放物線と比べる補助図."""
    d = np.linspace(-1.3 * radius, 1.3 * radius, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for b, c in zip((1, 2, 4, 8), ("tab:red", "tab:orange", "tab:green", "tab:blue")):
        levels = radius * (2.0 * (np.arange(2 ** b) + 0.5) / 2 ** b - 1.0)
        F = (d[:, None] >= levels[None, :]).mean(axis=1)
        axes[0].step(d, F, where="post", color=c, lw=1.2, label=f"b = {b}")
        axes[1].step(d, 2 * F * (1 - F), where="post", color=c, lw=1.2, label=f"b = {b}")
    Fc = np.clip((d + radius) / (2 * radius), 0, 1)
    axes[0].plot(d, Fc, "k--", lw=1, label="continuous")
    axes[1].plot(d, 2 * Fc * (1 - Fc), "k--", lw=1, label="continuous")
    axes[0].set_xlabel("pre-activation d")
    axes[0].set_ylabel("F(d)")
    axes[1].set_xlabel("pre-activation d")
    axes[1].set_ylabel(r"$\bar\phi(d) = 2F(1-F)$")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(description="§7.2 discrete (b-bit / LFSR) uniform noise (Fig. 7).")
    add_common_args(p)
    p.add_argument("--bits", type=str, default=",".join(str(b) for b in BITS))
    p.add_argument("--gens", type=str, default="iid,lfsr")
    p.add_argument("--grad-draws", type=int, default=32)
    args = finalize_args(p.parse_args(), default_out="out/fncl7_2_2")
    bits_list = [int(b) for b in args.bits.split(",")]
    gens = args.gens.split(",")
    if args.quick:
        bits_list, gens, args.grad_draws = [2], ["iid", "lfsr"], 4
    device = torch.device(args.device)
    noise = "uniform"
    response_figure(args.radius, args.out_dir / "fig_discrete_response.png")

    x_raw, target, x, t = make_task(device)
    conds = [("cont", 0)] + [(g, b) for g in gens for b in bits_list]
    results = {"config": config_dict(args), "methods": [m for m, _ in METHODS], "runs": {}}
    for gen, b in conds:
        key = "cont" if gen == "cont" else f"{gen}-{b}"
        rec = {"final_mse": {m: {} for m, _ in METHODS}, "fidelity": {}}
        for seed in args.seed_list:
            for name, spec in METHODS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh0 = model_factory(noise, args, device)
                fresh = (lambda f=fresh0, g=gen, bb=b, s=seed: configure(f(), g, bb, s))
                losses, pred, _ = run_method(spec, fresh, x, t, noise, args)
                mse = float(np.mean((pred - target) ** 2))
                rec["final_mse"][name][seed] = mse
                print(f"[{key}] seed {seed} {name:18s} final MSE = {mse:.5f}", flush=True)
            if seed == args.seed_list[0]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = configure(model_factory(noise, args, device)(), gen, b, seed)
                rec["fidelity"] = fidelity_untrained(net, x, t, draws=args.grad_draws)
                print(f"[{key}] untrained fidelity: "
                      + " ".join(f"cos({k})={c:.3f}" for k, c in rec["fidelity"]["cos"].items())
                      + f" | mirror r w1={rec['fidelity']['mirror_r']['w1']:.4f}", flush=True)
        results["runs"][key] = rec
        save_json(args.out_dir / "results.json", results)

    names = [m for m, _ in METHODS]
    lines = ["| noise | " + " | ".join(names) + " | cos(w0) | cos(w1) | r(w1) |",
             "|---" * (len(names) + 4) + "|"]
    for key, rec in results["runs"].items():
        cells = []
        for m in names:
            mu, sd = mean_std(rec["final_mse"][m].values())
            cells.append(f"{mu:.5f} ± {sd:.5f}")
        cos = rec["fidelity"]["cos"]
        lines.append(f"| {key} | " + " | ".join(cells)
                     + f" | {cos['w0']:.3f} | {cos['w1']:.3f}"
                     f" | {rec['fidelity']['mirror_r']['w1']:.4f} |")
    print("\n" + "\n".join(lines) + "\n")   # 表は標準出力のみ (ファイル保存しない)


if __name__ == "__main__":
    main()
