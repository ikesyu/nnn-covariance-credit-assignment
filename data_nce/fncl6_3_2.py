"""
fncl6_3_2.py — 論文 §6.3 Table 4 の最右列: MNIST 4 隠れ層での更新方向の cosine
(cov_jac / FA / DFA 対 backprop 参照)

§6.4 と同じ 784-256-...-256-10 の NNN (交差エントロピー; データは §6.4 とは別の
250 枚の class-balanced 部分集合) で、同一の forward サンプル・
同一の KDE 局所傾き・同一の読み出し誤差 (softmax - onehot) を使い、隠れ層 credit の
作り方だけを変えた更新方向を backprop 参照 (autograd) と層別の cosine で比べる:

  cov_jac : 共分散 weight mirror W_hat の再帰 (§4.3)
  fa      : 固定ランダム行列 B の再帰
  dfa     : 出力誤差を固定ランダム行列 B_l で直接投影

未学習の状態と、backprop (Adam) で --pretrain-epochs 学習した状態の両方で測る
(Table 4 は事前学習後、本文は未学習でも同じ結果になることを述べる)。
どちらの推定も --grad-draws 回の確率的 forward で平均してから比較する。

生成物 (out/fncl6_3_2/):
  results.json   -> 深さ x 状態 x 手法 x 層の cosine とノルム比
  (標準出力)     -> 層別の数表と、隠れ層にわたる cosine の範囲 (Table 4 の表記)

実行例:
  python data_nce/fncl6_3_2.py --device cuda          # 既定: 4 隠れ層
  python data_nce/fncl6_3_2.py --depths 2,4 --device cuda
  python data_nce/fncl6_3_2.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fncl_common import cosine, norm_ratio, save_json, fncl


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.3: update-direction cosine of cov_jac / FA / DFA on the MNIST NNN.")
    p.add_argument("--depths", type=str, default="4", help="カンマ区切りの隠れ層数")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=256, help="MNIST 部分集合の枚数 (10 で割った商 x 10 枚を使う: 既定 256 -> 250 枚)")
    p.add_argument("--num-samples", type=int, default=64, help="T")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--grad-draws", type=int, default=32)
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mnist-dir", type=str, default=None,
                   help="MNIST の保存先 (既定: data/mnist; 無ければダウンロード)")
    p.add_argument("--out", type=str, default="out/fncl6_3_2")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.width, args.n_inputs = "2", 32, 64
        args.num_samples, args.grad_draws, args.pretrain_epochs = 16, 4, 20
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    x_np, y_np = fncl.load_mnist(args.n_inputs, mnist_dir=args.mnist_dir)
    x = torch.tensor(x_np, device=device)
    labels = torch.tensor(y_np, device=device)

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        net0 = fncl.build_net(depth, args.width, args.sigma, args.crossing_h,
                              args.num_samples, device)
        init_state = {k: v.clone() for k, v in net0.state_dict().items()}
        run = {}
        for state in ("untrained", "pretrained"):
            net = fncl.build_net(depth, args.width, args.sigma, args.crossing_h,
                                 args.num_samples, device)
            net.load_state_dict(init_state)
            if state == "pretrained":
                fncl.pretrain_backprop(net, x, labels, args.pretrain_lr, args.pretrain_epochs)
                ce, acc, _ = fncl.evaluate(net, x, labels)
                print(f"[depth {depth}] pretrained CE={ce:.4f} acc={acc:.3f}", flush=True)
            B_fa = fncl.random_feedback(net, depth, "fa", args.width, device)
            B_dfa = fncl.random_feedback(net, depth, "dfa", args.width, device)

            def draw():
                with torch.no_grad():
                    s = fncl.collect_samples(net, x, passes=1)
                    W = fncl.measure_mirrors(s, depth)
                    out = {}
                    for tag, kw in (("cov_jac", {"W": W}), ("fa", {"B": B_fa}),
                                    ("dfa", {"B": B_dfa})):
                        for k, v in fncl.grads_from(net, x, labels, s, tag, **kw).items():
                            out[f"{tag}|{k}"] = v
                    return out

            g_auto = fncl.averaged(lambda: fncl.autograd_gradient(net, x, labels),
                                   args.grad_draws)
            g_est = fncl.averaged(draw, args.grad_draws)
            entry = {}
            for tag in ("cov_jac", "fa", "dfa"):
                fid = {k: {"cos": cosine(g_est[f"{tag}|{k}"], g_auto[k]),
                           "ratio": norm_ratio(g_est[f"{tag}|{k}"], g_auto[k])}
                       for k in g_auto}
                entry[tag] = fid
                print(f"[depth {depth} | {state}] {tag:8s} "
                      + "  ".join(f"{k}:{v['cos']:.3f}" for k, v in fid.items()), flush=True)
            run[state] = entry
        results["runs"][str(depth)] = run

    lines = ["| depth | state | layer | cov_jac cos | fa cos | dfa cos |",
             "|---|---|---|---|---|---|"]
    for depth, run in results["runs"].items():
        for state, entry in run.items():
            for k in entry["cov_jac"]:
                lines.append(f"| {depth} | {state} | {k} | "
                             f"{entry['cov_jac'][k]['cos']:.4f} | "
                             f"{entry['fa'][k]['cos']:.4f} | "
                             f"{entry['dfa'][k]['cos']:.4f} |")
    lines += ["", "| depth | state | method | hidden-layer cosine (min to max) |",
              "|---|---|---|---|"]
    for depth, run in results["runs"].items():
        for state, entry in run.items():
            for tag in ("cov_jac", "fa", "dfa"):
                cs = [v["cos"] for k, v in entry[tag].items() if k != "wout"]
                lines.append(f"| {depth} | {state} | {tag} | {min(cs):.2f} to {max(cs):.2f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)   # 表は標準出力のみ
    save_json(out_dir / "results.json", results)


if __name__ == "__main__":
    main()
