"""
fncl6_1_fig.py — §6.1 の図 (Fig. 6, 感度解析) を実行済みデータ (out/fncl6_1/results_*.json)
から再生成する。学習の再実行は不要。

fncl6_1.py が掃引ごとに保存する results_T.json, results_sigma.json, results_radius.json,
results_h.json, results_eps.json を読み、
  fig_rev_sweeps.png / .pdf
    (a)-(e) 各掃引に対する最終 MSE (mean ± std, log)
    (f)     T に対する未学習状態の勾配 cosine (層別) と weight mirror の r
を描く。存在しない掃引のパネルは非表示にする。

実行例:
  python data_nce/fncl6_1_fig.py                # 既定: --data out/fncl6_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# カテゴリカルパレット (Fig. 6, Fig. 7 共通)
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
SPECS = [("T", "ensemble size $T$"), ("sigma", r"Gaussian noise $\sigma$"),
         ("radius", "uniform noise half-width $r$"), ("h", "slope bandwidth $h$"),
         ("eps", r"regularizer $\epsilon$")]
SYMBOL = {"T": "$T$", "sigma": r"$\sigma$", "radius": "$r$", "h": "$h$", "eps": r"$\epsilon$"}


def load(path: Path):
    return json.load(open(path)) if path.exists() else None


def mean_std(d: dict):
    v = np.array(list(d.values()), dtype=float)
    return v.mean(), v.std()


def done_values(res):
    """results は掃引値ごとに逐次保存されるので、完了した値だけを返す."""
    return [v for v in res["values"] if str(v) in res["runs"]]


def plot_sweep(ax, res, xlabel):
    vals = done_values(res)
    xs = [float(v) for v in vals]
    for m in res["methods"]:
        mu, sd = zip(*[mean_std(res["runs"][str(v)]["final_mse"][m]) for v in vals])
        st = STYLE[m]
        ax.errorbar(xs, mu, yerr=sd, color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=4.5, lw=1.5, capsize=2, label=st["label"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("final MSE")


def fig_sweeps(data_dir: Path, formats) -> None:
    plt.rcParams.update(RC)
    loaded = [(k, lab, load(data_dir / f"results_{k}.json")) for k, lab in SPECS]
    if all(r is None for *_, r in loaded):
        raise SystemExit(f"{data_dir} に results_*.json が無い。fncl6_1.py を先に実行すること。")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()
    for i, (ax, (k, lab, res)) in enumerate(zip(axes, loaded)):
        if res is None:
            ax.set_visible(False)
            continue
        plot_sweep(ax, res, lab)
        if k in ("radius", "h", "sigma"):
            vals = [float(v) for v in done_values(res)]
            ax.set_xticks(vals)
            ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=7.5)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"({'abcde'[i]}) sweep over {SYMBOL[k]}", loc="left")
    # (f) T に対する勾配忠実度 (未学習)
    ax = axes[5]
    res = loaded[0][2]
    if res is not None:
        vals = done_values(res)
        xs = [float(v) for v in vals]
        fid = [res["runs"][str(v)]["fidelity"] for v in vals]
        for key, col, lab in (("w0", C["blue"], "cos, layer 1"),
                              ("w1", C["aqua"], "cos, layer 2"),
                              ("wout", C["violet"], "cos, readout")):
            ax.plot(xs, [f["cos"][key] for f in fid], color=col, marker="o", ms=4.5,
                    lw=1.5, label=lab)
        ax.plot(xs, [f["mirror_r"]["w1"] for f in fid], color=C["orange"], marker="D",
                ms=4.5, lw=1.5, ls=":", label="weight-mirror $r$, layer 2")
        ax.set_xscale("log")
        ax.set_xlabel("ensemble size $T$")
        ax.set_ylabel("cosine / Pearson $r$ (untrained)")
        ax.set_ylim(0.9, 1.003)
        ax.set_title("(f) gradient fidelity vs $T$", loc="left")
        ax.legend(fontsize=7.5, loc="lower right")
    else:
        ax.set_visible(False)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for fmt in formats:
        path = data_dir / f"fig_rev_sweeps.{fmt}"
        fig.savefig(path, dpi=160)
        print(f"  saved {path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="redraw Fig. 6 (§6.1 sweeps) from saved data")
    p.add_argument("--data", type=str, default="out/fncl6_1",
                   help="results_<sweep>.json のあるディレクトリ (出力先も同じ)")
    p.add_argument("--formats", type=str, default="png,pdf")
    args = p.parse_args()
    fig_sweeps(Path(args.data), [f.strip() for f in args.formats.split(",")])


if __name__ == "__main__":
    main()
