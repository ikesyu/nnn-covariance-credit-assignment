"""
fncl7_2_2_fig.py — §7.2 の図 (Fig. 7, 離散一様ノイズ) を実行済みデータ
(out/fncl7_2_2/results.json) から再生成する。学習の再実行は不要。

  fig_rev_discrete_full.png / .pdf
    (a) 独立な b ビット量子化一様ノイズの期待応答 phi_bar = 2F(1-F) の階段と連続の放物線
    (b) 16 ビット LFSR の上位 b ビットに対する最終 MSE (mean ± std, 3 seed)。
        細い水平線は連続一様ノイズの結果

実行例:
  python data_nce/fncl7_2_2_fig.py                # 既定: --data out/fncl7_2_2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# カテゴリカルパレット (Fig. 6, Fig. 7 共通; fncl6_1_fig.py と同じ)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
     "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9"}
STYLE = {
    "backprop":          dict(color=C["ink"], ls="--", marker="s", label="backprop"),
    "cov_deriv_kde":     dict(color=C["yellow"], ls="-", marker="^", label="cov_deriv"),
    "cov_jac_adam":      dict(color=C["blue"], ls="-", marker="o", label="cov_jac"),
    "cov_jac_full_adam": dict(color=C["orange"], ls="-", marker="D", label="cov_jac_full"),
}
RC = {"font.size": 9, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
      "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "xtick.color": C["ink"],
      "ytick.color": C["ink"], "legend.frameon": False}


def mean_std(d: dict):
    v = np.array(list(d.values()), dtype=float)
    return v.mean(), v.std()


def fig_discrete_full(data_dir: Path, formats, gen: str = "lfsr") -> None:
    plt.rcParams.update(RC)
    path = data_dir / "results.json"
    if not path.exists():
        raise SystemExit(f"{path} が無い。fncl7_2_2.py を先に実行すること。")
    res = json.load(open(path))
    radius = float(res["config"].get("radius", 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    # (a) 階段応答
    ax = axes[0]
    d = np.linspace(-1.3 * radius, 1.3 * radius, 2001)
    for b, c in zip((1, 2, 4, 8), (C["red"], C["orange"], C["aqua"], C["blue"])):
        levels = radius * (2.0 * (np.arange(2 ** b) + 0.5) / 2 ** b - 1.0)
        F = (d[:, None] >= levels[None, :]).mean(axis=1)
        ax.step(d, 2 * F * (1 - F), where="post", color=c, lw=1.3, label=f"$b = {b}$")
    Fc = np.clip((d + radius) / (2 * radius), 0, 1)
    ax.plot(d, 2 * Fc * (1 - Fc), color=C["ink"], ls="--", lw=1, label="continuous")
    ax.set_xlabel("pre-activation $d$")
    ax.set_ylabel(r"expected response $\bar\phi(d) = 2F(1-F)$")
    ax.set_title("(a) $b$-bit uniform noise: staircase response", loc="left", fontsize=9)
    ax.legend(fontsize=7.5)
    # (b) LFSR のビット数に対する最終 MSE
    cont = res["runs"]["cont"]["final_mse"]
    ax = axes[1]
    bits = sorted(int(k.split("-")[1]) for k in res["runs"] if k.startswith(gen + "-"))
    for m in res["methods"]:
        st = STYLE[m]
        mu, sd = zip(*[mean_std(res["runs"][f"{gen}-{b}"]["final_mse"][m]) for b in bits])
        ax.errorbar(bits, mu, yerr=sd, color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=4.5, lw=1.5, capsize=2, label=st["label"])
        ax.axhline(mean_std(cont[m])[0], color=st["color"], lw=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bits)
    ax.set_xticklabels([str(b) for b in bits])
    ax.set_yscale("log")
    ax.set_xlabel("bits per noise sample $b$")
    ax.set_title("(b) 16-bit LFSR, top $b$ bits", loc="left", fontsize=9)
    ax.set_ylabel("final MSE (thin lines: continuous noise)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    for fmt in formats:
        out = data_dir / f"fig_rev_discrete_full.{fmt}"
        fig.savefig(out, dpi=160)
        print(f"  saved {out}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="redraw Fig. 7 (§7.2 discrete noise) from saved data")
    p.add_argument("--data", type=str, default="out/fncl7_2_2")
    p.add_argument("--formats", type=str, default="png,pdf")
    p.add_argument("--gen", type=str, default="lfsr", help="パネル (b) に描く生成器 (lfsr / iid)")
    args = p.parse_args()
    fig_discrete_full(Path(args.data), [f.strip() for f in args.formats.split(",")], args.gen)


if __name__ == "__main__":
    main()
