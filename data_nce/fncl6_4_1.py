"""
fncl6_4_1.py — 論文 §6.4「Deeper Networks, Multiclass Classification, and the Readout
Estimator」Table 5(a): 深さ依存と noise field

MNIST の class-balanced な部分集合 (学習 1000 枚 / テスト 1000 枚, 固定 seed で 1 回
だけ抽出, 画素は [-1, 1]) 上の 784-256-...-256-10 の NNN (交差エントロピー, T=64,
ガウス sigma=0.5, Adam lr=1e-3, フルバッチ, 400 epoch) を、隠れ層数 2, 3, 4 と
seed 0-2 について次の 3 通りで学習する:

  backprop     : autograd + Adam の参照
  cov_jac      : 解析的な読み出し誤差 (softmax - onehot) つきの cov_jac
                 (共分散 weight mirror + 再帰 + KDE slope, EMA + Kolen-Pollack 追従)
  noise_field  : cov_jac に、周期 P=16 でサンプル軸にタイルした上流ノイズと位相クラス内
                 中心化による weight mirror の測定 (1 epoch に 1 層ずつローテーション,
                 残りは Kolen-Pollack 追従) を組み込んだもの

eval_every ごとに train CE / accuracy / test accuracy、学習時 weight mirror と真の
重みの Pearson r、更新方向と backprop 参照の層別 cosine を記録する。
Table 5(a) は最終 test accuracy (mean ± std) と、隠れ層にわたる cosine の最小値の
seed 平均。本文の「deep weight mirrors」の最小 r は、第 1 隠れ層を回帰子とする
mirror を除いた最小値の seed 平均である。

生成物 (out/fncl6_4_1/):
  results.json   -> runs[depth][seed][arm] = 学習履歴
  (標準出力)     -> Table 5(a) 形式の数表と weight mirror の r

実行例:
  python data_nce/fncl6_4_1.py --device cuda          # 3 深さ x 3 seed x 3 手法
  python data_nce/fncl6_4_1.py --depths 4 --seeds 0 --device cuda
  python data_nce/fncl6_4_1.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fncl_common import save_json, fncl

ARMS = ["backprop", "cov_jac", "noise_field"]


def min_hidden_cos(entry: dict) -> float:
    """隠れ層 (読み出し層を除く) にわたる cosine の最小値."""
    return min(v["cos"] for k, v in entry["fidelity"].items() if k != "wout")


def min_deep_mirror_r(entry: dict) -> float:
    """第 1 隠れ層を回帰子とする mirror (w1) を除いた weight mirror の r の最小値."""
    return min(v for k, v in entry["mirror_r"].items() if k != "w1")


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.4 Table 5(a): MNIST depth dependence of backprop / cov_jac / "
                    "cov_jac + noise field.")
    p.add_argument("--depths", type=str, default="2,3,4", help="カンマ区切りの隠れ層数")
    p.add_argument("--seeds", type=str, default="0,1,2", help="カンマ区切りの学習 seed")
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=64, help="T")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--period", type=int, default=16, help="noise field の周期 P (T を割り切ること)")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--grad-draws", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mnist-dir", type=str, default=None,
                   help="MNIST の保存先 (既定: data/mnist; 無ければダウンロード)")
    p.add_argument("--out", type=str, default="out/fncl6_4_1")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.depths, args.seeds, args.width = "2", "0", 32
        args.n_inputs, args.n_test = 64, 40
        args.num_samples, args.epochs, args.period = 16, 20, 8
        args.eval_every, args.grad_draws = 10, 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    log = lambda m: print(m, flush=True)  # noqa: E731

    x, labels, test = fncl.load_split(args.n_inputs, args.n_test, device, args.mnist_dir)
    log(f"train {x.shape[0]} / test {test[0].shape[0]}")

    def build():
        return fncl.build_net(depth, args.width, args.sigma, args.crossing_h,
                              args.num_samples, device)

    results = {"config": vars(args), "runs": {}}
    for depth in [int(d) for d in args.depths.split(",")]:
        results["runs"][str(depth)] = {}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            init_state = {k: v.clone() for k, v in build().state_dict().items()}
            run = {}
            for arm in arms:
                log(f"=== depth {depth} | seed {seed} : {arm} ===")
                torch.manual_seed(seed)          # 全 arm が同一初期重みから出発する
                net = build()
                net.load_state_dict(init_state)
                if arm == "backprop":
                    run[arm] = fncl.train_backprop_ce(net, x, labels, args.lr, args.epochs,
                                                      args.eval_every, log, test=test)
                elif arm == "cov_jac":
                    run[arm] = fncl.train_cov_jac_ce(net, x, labels, args.lr, args.epochs,
                                                     args.jac_ema, args.eval_every,
                                                     args.grad_draws, log, test=test)
                elif arm == "noise_field":
                    run[arm] = fncl.train_cov_jac_ce_noisefield(
                        net, x, labels, args.lr, args.epochs, args.jac_ema, args.eval_every,
                        args.grad_draws, log, test=test, period=args.period)
                else:
                    raise ValueError(arm)
            results["runs"][str(depth)][str(seed)] = run
            save_json(out_dir / "results.json", results)

    # ---- Table 5(a) ----
    lines = ["| hidden layers | backprop accuracy | cov_jac accuracy | cov_jac cosine | "
             "cov_jac + noise field accuracy | cov_jac + noise field cosine |",
             "|---|---|---|---|---|---|"]
    rlines = ["| hidden layers | cov_jac min deep weight-mirror r (end, seed mean) | "
              "cov_jac + noise field min deep weight-mirror r (end, seed mean) |", "|---|---|---|"]
    for depth, per_seed in results["runs"].items():
        cells, rcells = [], []
        for arm in ("backprop", "cov_jac", "noise_field"):
            ends = [per_seed[s][arm][-1] for s in per_seed if arm in per_seed[s]]
            if not ends:
                cells += ["--"] if arm == "backprop" else ["--", "--"]
                if arm != "backprop":
                    rcells.append("--")
                continue
            acc = [e["test_acc"] for e in ends]
            cells.append(f"{np.mean(acc):.3f} ± {np.std(acc):.3f}")
            if arm != "backprop":
                cells.append(f"{np.mean([min_hidden_cos(e) for e in ends]):.2f}")
                rcells.append(f"{np.mean([min_deep_mirror_r(e) for e in ends]):.3f}")
        lines.append(f"| {depth} | " + " | ".join(cells) + " |")
        rlines.append(f"| {depth} | " + " | ".join(rcells) + " |")
    print("\nTable 5(a): final test accuracy (mean ± std over seeds) and minimum hidden-layer "
          "cosine (seed mean)\n" + "\n".join(lines) + "\n\n" + "\n".join(rlines) + "\n",
          flush=True)   # 表は標準出力のみ (ファイル保存しない)


if __name__ == "__main__":
    main()
