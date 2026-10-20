"""
fncl6_4_2.py — 論文 §6.4 Table 5(b): 交差エントロピー下の読み出し誤差推定量

Table 5(a) と同じ MNIST 部分集合 (学習 1000 / テスト 1000) と 2 隠れ層 (784-256-256-10,
T=64, Adam lr=1e-3, 400 epoch) で、隠れ層 credit は全 arm とも cov_jac の weight mirror
再帰とし、読み出し誤差 e = dL/dy の作り方だけを差し替える:

  analytic : softmax(y_bar) - onehot (cov_jac; 解析的な損失微分)
  cov      : 出力ユニットごとの単変量回帰 Cov_T(L, y_k) / Var_T(y_k) (Eq. 23, 補正なし)
  cov_m3   : 同上から 3 次中心モーメントを引く (cov_jac_full; 二乗損失でのみ厳密な補正)
  mv_cov   : 10 出力の多変量回帰 Cov(y, y)^{-1} Cov(y, L) (Eq. 26; ridge は平均読み出し
             分散の 1e-3 倍, モーメント補正なし)
  probe    : 対称ガウスプローブ xi (alpha = 0.2) を logit サンプルに加え
             Cov_T(L(y + xi), xi_k) / Var_T(xi_k) (Stein 型)

L は各サンプル logit に対する per-sample 交差エントロピー。eval_every ごとに
test accuracy と、推定した e と解析形の cosine (Table 5(b) の最右列) を記録する。
--arms に backprop を加えると autograd + Adam の参照も同じ初期重みで走る。

生成物 (out/fncl6_4_2/):
  results.json   -> runs[seed][arm] = 学習履歴
  (標準出力)     -> Table 5(b) 形式の数表

実行例:
  python data_nce/fncl6_4_2.py --device cuda          # 5 推定量 x 3 seed
  python data_nce/fncl6_4_2.py --quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fncl_common import cosine, save_json, fncl

ARMS = ["analytic", "cov", "cov_m3", "mv_cov", "probe"]
ARM_LABEL = {"analytic": "analytical gradient softmax(y) - t (cov_jac)",
             "cov": "Eq. (23), uncorrected",
             "cov_m3": "Eq. (23) with correction (cov_jac_full)",
             "mv_cov": "Eq. (26), multivariate, uncorrected",
             "probe": "symmetric probe, Gaussian, alpha = 0.2",
             "backprop": "backprop (reference)"}


def readout_error(arm: str, s: dict, labels: torch.Tensor, probe_alpha: float,
                  mv_ridge: float) -> torch.Tensor:
    """読み出し誤差の推定 e_hat ~= dL/dy [N, 10] を arm に従って作る."""
    ys = s["ys"]                                                   # [N, T, 10]
    if arm == "analytic":
        return fncl.ce_error(s["y"], labels)
    if arm == "probe":
        xi = probe_alpha * torch.randn_like(ys)
        L = fncl.per_sample_ce(ys + xi, labels)                    # [N, T]
        cxi = xi - xi.mean(dim=1, keepdim=True)
        cL = (L - L.mean(dim=1, keepdim=True)).unsqueeze(-1)
        return (cL * cxi).mean(dim=1) / ((cxi ** 2).mean(dim=1) + fncl.EPS)
    L = fncl.per_sample_ce(ys, labels)
    cy = ys - ys.mean(dim=1, keepdim=True)                         # [N, T, 10]
    cL = (L - L.mean(dim=1, keepdim=True)).unsqueeze(-1)           # [N, T, 1]
    cov_Ly = (cL * cy).mean(dim=1)                                 # [N, 10]
    if arm == "mv_cov":
        T = ys.shape[1]
        cov_yy = torch.einsum("ntk,nti->nki", cy, cy) / T          # [N, 10, 10]
        scale = cov_yy.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1) + 1e-12
        eye = torch.eye(cov_yy.shape[1], device=ys.device)
        return torch.linalg.solve(cov_yy + mv_ridge * scale * eye,
                                  cov_Ly.unsqueeze(-1)).squeeze(-1)
    if arm == "cov_m3":
        cov_Ly = cov_Ly - (cy ** 3).mean(dim=1)                    # 3 次モーメント補正
    return cov_Ly / ((cy ** 2).mean(dim=1) + fncl.EPS)


def train_arm(net, x, labels, arm: str, lr: float, epochs: int, jac_ema: float,
              eval_every: int, probe_alpha: float, mv_ridge: float, log, test):
    """cov_jac の隠れ層 credit に、arm の読み出し誤差推定を組み合わせて学習する."""
    n_hidden = len(fncl.crossing_layers(net))
    optim = fncl.ManualOpt("adam")
    W_ema, history = {}, []
    for epoch in range(epochs):
        with torch.no_grad():
            s = fncl.collect_samples(net, x, passes=1)
            meas = fncl.measure_mirrors(s, n_hidden)
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * v
            e = readout_error(arm, s, labels, probe_alpha, mv_ridge)          # [N, 10]
            gW, gb = fncl.manual_grads(net, x, labels, W_ema, s, e=e)
            steps = fncl.apply_updates(net, optim, gW, gb, lr, n_hidden)
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]
        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = fncl.evaluate(net, x, labels, *test)
            with torch.no_grad():
                ss = fncl.collect_samples(net, x, passes=1)
                e_hat = readout_error(arm, ss, labels, probe_alpha, mv_ridge)
                e_true = fncl.ce_error(ss["y"], labels)
                r_cos = cosine(e_hat, e_true)
                r_ratio = float(e_hat.norm() / (e_true.norm() + 1e-12))
            mr = fncl.mirror_pearson(net, W_ema, n_hidden)
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te_acc,
                            "readout_cos": r_cos, "readout_norm_ratio": r_ratio,
                            "mirror_r": mr})
            log(f"  [{arm} ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} test={te_acc:.3f} | "
                f"readout cos={r_cos:.3f} ratio={r_ratio:.2f} | mirror "
                + " ".join(f"{k}={v:.3f}" for k, v in mr.items()))
    return history


def main() -> None:
    p = argparse.ArgumentParser(
        description="§6.4 Table 5(b): readout-error estimators under the cross-entropy loss.")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help="カンマ区切り (backprop を加えると参照も走る)")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-inputs", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--crossing-h", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--jac-ema", type=float, default=0.9)
    p.add_argument("--probe-alpha", type=float, default=0.2)
    p.add_argument("--mv-ridge", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mnist-dir", type=str, default=None,
                   help="MNIST の保存先 (既定: data/mnist; 無ければダウンロード)")
    p.add_argument("--out", type=str, default="out/fncl6_4_2")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.width, args.n_inputs, args.n_test = "0", 32, 64, 36
        args.num_samples, args.epochs, args.eval_every = 16, 20, 10
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    log = lambda m: print(m, flush=True)  # noqa: E731

    x, labels, test = fncl.load_split(args.n_inputs, args.n_test, device, args.mnist_dir)
    log(f"train {x.shape[0]} / test {test[0].shape[0]}, depth {args.depth}")

    results = {"config": vars(args), "runs": {}}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net0 = fncl.build_net(args.depth, args.width, args.sigma, args.crossing_h,
                              args.num_samples, device)
        init_state = {k: v.clone() for k, v in net0.state_dict().items()}

        def fresh():
            torch.manual_seed(seed)
            n = fncl.build_net(args.depth, args.width, args.sigma, args.crossing_h,
                               args.num_samples, device)
            n.load_state_dict(init_state)
            return n

        run = {}
        for arm in arms:
            log(f"=== seed {seed} | readout error: {arm} ===")
            if arm == "backprop":
                run[arm] = fncl.train_backprop_ce(fresh(), x, labels, args.lr, args.epochs,
                                                  args.eval_every, log, test=test)
            else:
                run[arm] = train_arm(fresh(), x, labels, arm, args.lr, args.epochs,
                                     args.jac_ema, args.eval_every, args.probe_alpha,
                                     args.mv_ridge, log, test)
            results["runs"][str(seed)] = run
            save_json(out_dir / "results.json", results)

    lines = ["| readout error | test accuracy (mean ± std) | cosine to the analytical error (end) |",
             "|---|---|---|"]
    for arm in arms:
        ends = [results["runs"][str(s)][arm][-1] for s in seeds]
        acc = [e["test_acc"] for e in ends]
        cos = [e.get("readout_cos") for e in ends]
        c = f"{np.mean(cos):.2f}" if cos[0] is not None else "--"
        lines.append(f"| {ARM_LABEL.get(arm, arm)} | {np.mean(acc):.3f} ± {np.std(acc):.3f} | {c} |")
    print("\nTable 5(b)\n" + "\n".join(lines) + "\n", flush=True)   # 表は標準出力のみ


if __name__ == "__main__":
    main()
