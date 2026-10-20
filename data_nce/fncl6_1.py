"""
fncl6_1.py — 論文 §6.1「Sensitivity to the Ensemble Size, Noise Intensity, and
Estimator Hyperparameters」(Fig. 6)

§5.1 の sin(x) プロトコル (1-64-64-1, T=64, 1500 epoch, seed 0-2, 同一初期重み) を
保ったまま 1 つのハイパーパラメータだけを掃引し、backprop / cov_deriv (KDE, SGD) /
cov_jac (Adam) / cov_jac_full (Adam, 3 次モーメント補正) の最終 MSE を比較する:

  --sweep T       : アンサンブルサイズ T (Fig. 6(a))。加えて未学習状態の勾配 cosine と
                    weight mirror の Pearson r (§5.3 のプロトコル; Fig. 6(f))
  --sweep sigma   : ガウスノイズ std (Fig. 6(b))
  --sweep radius  : 一様ノイズ半幅 r (Fig. 6(c); --noise uniform を自動設定)
  --sweep h       : KDE slope の帯域幅 = 交差しきい値の半幅 (Fig. 6(d))
  --sweep eps     : 共分散／分散比の正則化 ε (Fig. 6(e); fncl.train.EPS を差し替える)

各掃引値で 3 seed の最終 MSE、評価 MSE が閾値を初めて切る epoch、学習曲線
(5 epoch 間引き)、先頭 seed の未学習状態の勾配 cosine と mirror r を記録する。
本文の「backprop の最終 MSE の 2 倍に達する epoch」は保存した学習曲線から求めて表示する。

生成物 (out/fncl6_1/):
  results_<sweep>.json   -> 掃引ごとの数値一式 (fncl6_1_fig.py が Fig. 6 を描く)
  (標準出力)             -> 掃引ごとの数表 (最終 MSE, cosine, r, 収束 epoch)

実行例:
  python data_nce/fncl6_1.py                          # 5 掃引すべて (時間がかかる)
  python data_nce/fncl6_1.py --sweep T --device cuda
  python data_nce/fncl6_1.py --sweep T,sigma --quick
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from fncl_common import (add_common_args, config_dict, epochs_to, fidelity_untrained,
                         finalize_args, make_task, mean_std, model_factory, run_method,
                         save_json)
import fncl.train as fncl_train

METHODS = [
    ("backprop",          {"kind": "backprop"}),
    ("cov_deriv_kde",     {"method": "cov_deriv", "slope": "kde"}),
    ("cov_jac_adam",      {"method": "cov_jac", "opt": "adam", "jac_track": True}),
    ("cov_jac_full_adam", {"method": "cov_jac_full", "opt": "adam",
                           "jac_track": True, "jac_out": "cov_m3"}),
]

SWEEPS = {
    "T":      [4, 8, 16, 32, 64, 128, 256],
    "sigma":  [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    "radius": [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0],
    "h":      [0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
    "eps":    [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
}
MSE_THRESHOLD = 0.002      # 収束速度の指標: 評価 MSE がこれを初めて下回る epoch
CURVE_STRIDE = 5           # 学習曲線の保存間隔 (epoch)


def apply_value(args, sweep: str, v):
    """掃引値 v を反映した args のコピーを返す (eps は fncl.train.EPS を差し替える)."""
    a = copy.deepcopy(args)
    if sweep == "T":
        a.num_samples = int(v)
    elif sweep == "sigma":
        a.sigma = float(v)
    elif sweep == "radius":
        a.radius = float(v)
    elif sweep == "h":
        a.crossing_h = float(v)
    elif sweep == "eps":
        fncl_train.EPS = float(v)
    return a


def run_sweep(sweep: str, args, values, methods, device) -> dict:
    noise = args.noise or ("uniform" if sweep == "radius" else "gaussian")
    eps_default = fncl_train.EPS
    results = {"config": config_dict(args), "sweep": sweep, "noise": noise,
               "values": values, "methods": [m for m, _ in methods], "runs": {}}
    for v in values:
        a = apply_value(args, sweep, v)
        x_raw, target, x, t = make_task(device)
        rec = {"final_mse": {m: {} for m, _ in methods},
               "epochs_to_thr": {m: {} for m, _ in methods},
               "curves": {m: {} for m, _ in methods}, "fidelity": {}}
        for seed in a.seed_list:
            for name, spec in methods:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh = model_factory(noise, a, device)
                losses, pred, _ = run_method(spec, fresh, x, t, noise, a)
                mse = float(np.mean((pred - target) ** 2))
                rec["final_mse"][name][seed] = mse
                rec["epochs_to_thr"][name][seed] = epochs_to(losses, MSE_THRESHOLD)
                rec["curves"][name][seed] = [float(c) for c in losses[::CURVE_STRIDE]]
                print(f"[{sweep}={v}] seed {seed} {name:18s} final MSE = {mse:.5f}",
                      flush=True)
            if seed == a.seed_list[0]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = model_factory(noise, a, device)()
                rec["fidelity"] = fidelity_untrained(net, x, t, draws=args.grad_draws)
                print(f"[{sweep}={v}] untrained fidelity: "
                      + " ".join(f"cos({k})={c:.3f}" for k, c in rec["fidelity"]["cos"].items())
                      + f" | mirror r w1={rec['fidelity']['mirror_r']['w1']:.4f}"
                      f" wout={rec['fidelity']['mirror_r']['wout']:.4f}", flush=True)
        results["runs"][str(v)] = rec
        fncl_train.EPS = eps_default
        save_json(args.out_dir / f"results_{sweep}.json", results)
    return results


def sweep_tables(results: dict) -> str:
    """最終 MSE (mean ± std)・未学習 cosine・mirror r と、収束 epoch の数表."""
    sweep, values, names = results["sweep"], results["values"], results["methods"]
    lines = [f"| {sweep} | " + " | ".join(names) + " | cos(w0) | cos(w1) | r(w1) |",
             "|---" * (len(names) + 4) + "|"]
    for v in values:
        rec = results["runs"][str(v)]
        cells = []
        for m in names:
            mu, sd = mean_std(rec["final_mse"][m].values())
            cells.append(f"{mu:.5f} ± {sd:.5f}")
        fid = rec.get("fidelity", {})
        cos = fid.get("cos", {})
        lines.append(f"| {v} | " + " | ".join(cells)
                     + f" | {cos.get('w0', float('nan')):.3f} | {cos.get('w1', float('nan')):.3f}"
                     f" | {fid.get('mirror_r', {}).get('w1', float('nan')):.4f} |")
    lines.append("")
    lines.append(f"| {sweep} | " + " | ".join(f"epochs to MSE<{MSE_THRESHOLD} ({m})"
                                               for m in names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for v in values:
        rec = results["runs"][str(v)]
        cells = []
        for m in names:
            vals = list(rec["epochs_to_thr"][m].values())
            n_reached = sum(1 for e in vals if e is not None)
            mu, _ = mean_std(vals)
            cells.append(f"{mu:.0f} ({n_reached}/{len(vals)})" if n_reached else "–")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    if "backprop" in names:
        # 本文 §6.1: backprop の終盤 20 epoch の平均 MSE の 2 倍に初めて達する epoch
        lines.append("")
        lines.append(f"| {sweep} | " + " | ".join(
            f"epochs to 2x backprop final ({m})" for m in names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for v in values:
            rec = results["runs"][str(v)]
            cells = []
            for m in names:
                es = []
                for seed, curve in rec["curves"][m].items():
                    bp = rec["curves"]["backprop"][seed]
                    thr = 2.0 * float(np.mean(bp[-4:]))
                    e = epochs_to(curve, thr)
                    es.append(None if e is None else e * CURVE_STRIDE)
                n_reached = sum(1 for e in es if e is not None)
                mu, _ = mean_std(es)
                cells.append(f"{mu:.0f} ({n_reached}/{len(es)})" if n_reached else "–")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.1 sensitivity sweeps on the sin(x) protocol (Fig. 6).")
    add_common_args(p)
    p.add_argument("--sweep", type=str, default=",".join(SWEEPS),
                   help="カンマ区切りの掃引名 (T, sigma, radius, h, eps)")
    p.add_argument("--values", type=str, default=None,
                   help="カンマ区切りで掃引値を上書き (掃引が 1 つのときのみ)")
    p.add_argument("--noise", choices=("gaussian", "uniform"), default=None,
                   help="既定: radius 掃引は uniform, それ以外は gaussian")
    p.add_argument("--grad-draws", type=int, default=32,
                   help="未学習状態の勾配推定の平均化 draw 数")
    p.add_argument("--methods", type=str, default=None,
                   help="カンマ区切りの手法名で METHODS を絞る")
    args = finalize_args(p.parse_args(), default_out="out/fncl6_1")
    sweeps = [s for s in args.sweep.split(",") if s]
    methods = METHODS if args.methods is None else [
        m for m in METHODS if m[0] in args.methods.split(",")]
    if args.values is not None and len(sweeps) != 1:
        raise SystemExit("--values は --sweep が 1 つのときだけ使える")
    if args.quick:
        args.grad_draws = 4
    device = torch.device(args.device)
    for sweep in sweeps:
        values = SWEEPS[sweep] if args.values is None else [
            float(s) for s in args.values.split(",")]
        if args.quick:
            values = values[:2]
        results = run_sweep(sweep, args, values, methods, device)
        print("\n" + sweep_tables(results) + "\n")   # 表は標準出力のみ (ファイル保存しない)


if __name__ == "__main__":
    main()
