"""
fncl6_3_3.py — 論文 §6.3「a Conventional Weight Mirror」: 専用フェーズの weight mirror との比較

同じネットワーク・データ・最適化 (§5.1 の sin(x) プロトコル, Adam, EMA, Kolen-Pollack
追従) で、weight mirror の測り方だけを差し替える (fncl.mirror.train_jac_flex):

  intrinsic        : 本論文の cov_jac。推論用 forward の揺らぎ (N*T サンプル) から推定
  probe_reg_K<k>   : 専用フェーズの weight mirror。各層の入力に等方ガウス摂動ベクトルを
                     K 本注入し、前活性への回帰で推定 (Akrout et al. 2019 の mirror mode に相当)
  probe_hebb_K<k>  : 同じ専用フェーズで Hebb 型 (非中心化) の推定
  intrinsic_nokp   : 対照。Kolen-Pollack 追従を切った intrinsic
  mv               : 参照。同じ揺らぎからの多変量回帰 (O(H^3))

記録: 最終 MSE (3 seed)、学習中の weight mirror の r (隠れ層 w1 と読み出し層 out)、
1 更新あたりの追加線形層評価数。本文は、最終 MSE の seed 平均の範囲、intrinsic の
学習中最小 r、K = 256 と K = 16 の学習中最小 r (隠れ層) を述べる。

生成物 (out/fncl6_3_3/):
  results.json   -> 数値一式 (最終 MSE, mirror r の履歴, 追加評価数)
  (標準出力)     -> 手法別の数表

実行例:
  python data_nce/fncl6_3_3.py --device cuda
  python data_nce/fncl6_3_3.py --quick
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fncl_common import (add_common_args, config_dict, finalize_args, make_task,
                         mean_std, model_factory, save_json, fncl)

ARMS = [
    ("intrinsic",        {"mirror": "intrinsic"}),
    ("intrinsic_nokp",   {"mirror": "intrinsic", "kp_track": False}),
    ("mv",               {"mirror": "mv"}),
    ("probe_reg_K4",     {"mirror": "probe_reg", "probe_k": 4}),
    ("probe_reg_K16",    {"mirror": "probe_reg", "probe_k": 16}),
    ("probe_reg_K64",    {"mirror": "probe_reg", "probe_k": 64}),
    ("probe_reg_K256",   {"mirror": "probe_reg", "probe_k": 256}),
    ("probe_hebb_K16",   {"mirror": "probe_hebb", "probe_k": 16}),
    ("probe_hebb_K64",   {"mirror": "probe_hebb", "probe_k": 64}),
    ("probe_hebb_K256",  {"mirror": "probe_hebb", "probe_k": 256}),
]


def main() -> None:
    p = argparse.ArgumentParser(description="§6.3 dedicated-phase weight-mirror baseline.")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--probe-alpha", type=float, default=0.1, help="摂動ベクトルの振幅")
    p.add_argument("--arms", type=str, default=None, help="カンマ区切りで arm を絞る")
    args = finalize_args(p.parse_args(), default_out="out/fncl6_3_3")
    arms = ARMS if args.arms is None else [a for a in ARMS if a[0] in args.arms.split(",")]
    if args.quick:
        arms = arms[:4]
    device = torch.device(args.device)
    x_raw, target, x, t = make_task(device)
    log_every = max(1, args.epochs // 5)

    results = {"config": config_dict(args), "runs": {}}
    for name, spec in arms:
        rec = {"final_mse": {}, "history": {}, "extra_evals_per_update": None}
        for seed in args.seed_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = model_factory(args.noise, args, device)()
            losses, hist, extra = fncl.train_jac_flex(
                net, x, t, args.lr, args.epochs, opt="adam", jac_ema=args.jac_ema,
                probe_alpha=args.probe_alpha, eval_every=max(1, args.epochs // 30),
                log_every=log_every if seed == args.seed_list[0] else 0, tag=name, **spec)
            pred = fncl.predict(net, x)
            mse = float(np.mean((pred - target) ** 2))
            rec["final_mse"][seed] = mse
            rec["history"][seed] = hist
            rec["extra_evals_per_update"] = extra / args.epochs
            print(f"[seed {seed}] {name:18s} final MSE = {mse:.5f}  "
                  f"(extra linear evals/update = {extra / args.epochs:.0f})", flush=True)
        results["runs"][name] = rec
        save_json(args.out_dir / "results.json", results)

    lines = ["| mirror source | final MSE (mean ± std) | mirror r at end (w1 / out, first seed) "
             "| min r during training (w1 / out, first seed) | min r (w1 / out, all seeds) "
             "| extra linear evals / update | dedicated phase |",
             "|---|---|---|---|---|---|---|"]
    for name, rec in results["runs"].items():
        mu, sd = mean_std(rec["final_mse"].values())
        h0 = rec["history"][args.seed_list[0]]
        r_end = h0[-1]["mirror_r"]
        r_min0 = {k: min(e["mirror_r"][k] for e in h0) for k in r_end}
        r_min = {k: min(e["mirror_r"][k] for h in rec["history"].values() for e in h)
                 for k in r_end}
        lines.append(f"| {name} | {mu:.5f} ± {sd:.5f} | {r_end['1']:.4f} / {r_end['out']:.4f}"
                     f" | {r_min0['1']:.4f} / {r_min0['out']:.4f}"
                     f" | {r_min['1']:.4f} / {r_min['out']:.4f}"
                     f" | {rec['extra_evals_per_update']:.0f}"
                     f" | {'no' if name.startswith(('intrinsic', 'mv')) else 'yes'} |")
    print("\n" + "\n".join(lines) + "\n")   # 表は標準出力のみ (ファイル保存しない)


if __name__ == "__main__":
    main()
