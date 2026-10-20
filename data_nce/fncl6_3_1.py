"""
fncl6_3_1.py — 論文 §6.3「Comparison with Alternative Forward-only Rules」(Table 4 の最終 MSE)

§5.1 の sin(x) プロトコル (1-64-64-1, T=64, 1500 epoch, ガウス sigma=0.5, seed 0-2,
同一初期重み) を変えずに、同じ NNN 上で代替 forward-only 学習則を走らせる:

  fa_{sgd,adam}    : feedback alignment。cov_jac の再帰の W_hat を固定ランダム行列に置換
  dfa_{sgd,adam}   : direct feedback alignment。出力誤差を固定ランダム行列で各隠れ層へ投影
  np_a<alpha>_*    : node perturbation。全ユニットに振幅 alpha の外部摂動を注入する
                     common-random-number 推定 (fncl.train の cov_deriv_gate_crn)
  pepita_f<scale>  : PEPITA。誤差変調した 2 回目の forward (変調係数 3 通り)

--tasks bench で §5.6 の 3 ベンチマーク (friedman1 / moons / circles) でも走らせる。
Table 4 に載せるのは FA (Adam), DFA (Adam), node perturbation (sin(x) は Adam,
ベンチマークは SGD, いずれも alpha = 0.1), PEPITA (sin(x) のみ, 3 係数の最良) で、
backprop と cov_jac の行は Table 2 (fncl5_2.py) と Table 3 (fncl5_6.py) の値である。
--with-reference を付けると backprop と cov_jac (Adam) も同じ行列で走らせる。

生成物 (out/fncl6_3_1/):
  results.json        -> sin(x) の最終 MSE (手法 x seed)
  results_bench.json  -> 3 ベンチマークの最終 MSE と分類精度
  (標準出力)          -> 手法別の数表と Table 4 形式の要約

実行例:
  python data_nce/fncl6_3_1.py --device cuda          # sin(x) と 3 ベンチマーク
  python data_nce/fncl6_3_1.py --tasks sin --quick
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fncl_common import (add_common_args, config_dict, finalize_args, make_task,
                         model_factory, mse_table_md, run_method, save_json, fncl)
from fncl5_6 import DATA_SEED, TASKS, accuracy, model_factory_nd

SIN_METHODS = [
    ("fa_sgd",       {"kind": "fa", "opt": "sgd"}),
    ("fa_adam",      {"kind": "fa", "opt": "adam"}),
    ("dfa_sgd",      {"kind": "dfa", "opt": "sgd"}),
    ("dfa_adam",     {"kind": "dfa", "opt": "adam"}),
    ("np_a0.1_sgd",  {"kind": "np", "alpha": 0.1, "opt": "sgd"}),
    ("np_a0.3_sgd",  {"kind": "np", "alpha": 0.3, "opt": "sgd"}),
    ("np_a0.1_adam", {"kind": "np", "alpha": 0.1, "opt": "adam"}),
    ("pepita_f0.1",  {"kind": "pepita", "fscale": 0.1}),
    ("pepita_f0.5",  {"kind": "pepita", "fscale": 0.5}),
    ("pepita_f1.0",  {"kind": "pepita", "fscale": 1.0}),
]
BENCH_METHODS = [
    ("fa_adam",      {"kind": "fa", "opt": "adam"}),
    ("dfa_adam",     {"kind": "dfa", "opt": "adam"}),
    ("np_a0.1_sgd",  {"kind": "np", "alpha": 0.1, "opt": "sgd"}),
    ("np_a0.3_sgd",  {"kind": "np", "alpha": 0.3, "opt": "sgd"}),
]
# --with-reference: 同じ行列で走らせる参照 (Table 2, 3 の backprop と cov_jac と同じ設定)
REFERENCE_METHODS = [
    ("backprop",     {"kind": "backprop"}),
    ("cov_jac_adam", {"method": "cov_jac", "opt": "adam", "jac_track": True}),
]
# Table 4 の行: (表示名, sin(x) の手法名, ベンチマークの手法名)
TABLE4_ROWS = [("FA", "fa_adam", "fa_adam"), ("DFA", "dfa_adam", "dfa_adam"),
               ("node perturbation", "np_a0.1_adam", "np_a0.1_sgd"),
               ("PEPITA", "pepita_best", None)]


def run_spec(spec: dict, fresh, x, t, args, log_every: int):
    """1 手法を学習して (losses, pred) を返す."""
    kind = spec.get("kind")
    if kind in ("fa", "dfa"):
        net = fresh()
        losses = fncl.train_fa(net, x, t, kind, args.lr, args.epochs,
                               opt=spec.get("opt", "adam"), log_every=log_every)
    elif kind == "pepita":
        net = fresh()
        losses = fncl.train_pepita(net, x, t, args.lr, args.epochs, fscale=spec["fscale"],
                                   opt=spec.get("opt", "sgd"), log_every=log_every)
    elif kind == "np":
        # node perturbation = common-random-number ゲートを全ユニットに適用
        net = fresh()
        losses, _ = fncl.train_cov(
            net, x, t, args.noise, args.sigma, args.radius, "cov_deriv_gate_crn",
            args.lr, args.epochs, credit="per_input", opt=spec.get("opt", "sgd"),
            gate_block_size=args.hidden_dim, gate_alpha=spec["alpha"],
            log_every=log_every)
    else:  # 参照 (backprop / cov_jac): fncl_common.run_method に委ねる
        losses, pred, _ = run_method(spec, fresh, x, t, args.noise, args, log_every)
        return losses, pred
    return losses, fncl.predict(net, x)


def summary_table(mse_sin: dict, bench: dict, seed_list) -> str:
    """Table 4 の最終 MSE 列 (代替手法の行) を本文の表記で要約する."""
    tasks = list(bench.keys())
    lines = ["| method | sin(x) | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 2) + "|"]

    def cell(per_seed, kind):
        mses = [per_seed[s]["mse"] if isinstance(per_seed[s], dict) else per_seed[s]
                for s in seed_list]
        c = f"{np.mean(mses):.5f} ± {np.std(mses):.5f}"
        if kind == "classification":
            accs = [per_seed[s]["acc"] for s in seed_list]
            if np.mean(accs) < 1.0:
                c += f" ({np.mean(accs):.3f})"
        return c

    for label, sin_name, bench_name in TABLE4_ROWS:
        if sin_name == "pepita_best":
            cands = {k: np.mean([v[s] for s in seed_list]) for k, v in mse_sin.items()
                     if k.startswith("pepita")}
            sin_name = min(cands, key=cands.get) if cands else None
        cells = [cell(mse_sin[sin_name], "regression") if sin_name in mse_sin else "--"]
        for tn in tasks:
            per = bench[tn]["per_method"].get(bench_name) if bench_name else None
            cells.append(cell(per, bench[tn]["kind"]) if per else "--")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("backprop and cov_jac rows of Table 4: Table 2 (fncl5_2.py) and Table 3 (fncl5_6.py).")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.3 alternative forward-only rules (FA / DFA / NP / PEPITA) on the NNN.")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    p.add_argument("--tasks", type=str, default="sin,bench",
                   help="sin: sin(x) 回帰, bench: §5.6 の 3 ベンチマーク (カンマ区切り)")
    p.add_argument("--with-reference", action="store_true",
                   help="backprop と cov_jac (Adam) も同じ行列で走らせる")
    args = finalize_args(p.parse_args(), default_out="out/fncl6_3_1")
    device = torch.device(args.device)
    log_every = max(1, args.epochs // 5)
    tasks = args.tasks.split(",")
    sin_methods = SIN_METHODS + (REFERENCE_METHODS if args.with_reference else [])
    bench_methods = BENCH_METHODS + (REFERENCE_METHODS if args.with_reference else [])

    mse, results = {}, {}
    if "sin" in tasks:
        x_raw, target, x, t = make_task(device)
        mse = {name: {} for name, _ in sin_methods}
        for seed in args.seed_list:
            for name, spec in sin_methods:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fresh = model_factory(args.noise, args, device)
                losses, pred = run_spec(spec, fresh, x, t, args, log_every)
                mse[name][seed] = float(np.mean((pred - target) ** 2))
                print(f"[seed {seed}] {name:16s} final MSE = {mse[name][seed]:.5f}", flush=True)
        print("\n" + mse_table_md(mse, args.seed_list,
                                  f"Alternative rules on sin(x), noise={args.noise}, "
                                  f"H={args.hidden_dim}, T={args.num_samples}, "
                                  f"epochs={args.epochs}, lr={args.lr}"))
        save_json(args.out_dir / "results.json", {"config": config_dict(args), "final_mse": mse})

    if "bench" in tasks:
        for task_name, make in TASKS.items():
            x_np, t_np, kind = make(np.random.default_rng(DATA_SEED))
            x = torch.tensor(x_np, device=device)
            t = torch.tensor(t_np, device=device).unsqueeze(1)
            results[task_name] = {"kind": kind, "per_method": {}}
            for seed in args.seed_list:
                for name, spec in bench_methods:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    fresh = model_factory_nd(args.noise, x_np.shape[1], args, device)
                    losses, pred = run_spec(spec, fresh, x, t, args, log_every)
                    entry = {"mse": float(np.mean((pred - t_np) ** 2))}
                    if kind == "classification":
                        entry["acc"] = accuracy(pred, t_np)
                    results[task_name]["per_method"].setdefault(name, {})[seed] = entry
                    extra = f"  acc = {entry['acc']:.3f}" if kind == "classification" else ""
                    print(f"[{task_name} | seed {seed}] {name:16s} "
                          f"final MSE = {entry['mse']:.5f}{extra}", flush=True)
        task_names = list(results.keys())
        lines = ["| method | " + " | ".join(task_names) + " |",
                 "|---" * (len(task_names) + 1) + "|"]
        for name, _ in bench_methods:
            cells = []
            for tn in task_names:
                per_seed = results[tn]["per_method"][name]
                mses = [per_seed[s]["mse"] for s in args.seed_list]
                c = f"{np.mean(mses):.5f} ± {np.std(mses):.5f}"
                if results[tn]["kind"] == "classification":
                    accs = [per_seed[s]["acc"] for s in args.seed_list]
                    c += f" ({np.mean(accs):.3f})"
                cells.append(c)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print("\n" + "\n".join(lines) + "\n")
        save_json(args.out_dir / "results_bench.json",
                  {"config": config_dict(args), "data_seed": DATA_SEED, "results": results})

    if mse and results:
        print("\nTable 4 (final MSE columns):\n" + summary_table(mse, results, args.seed_list) + "\n")


if __name__ == "__main__":
    main()
