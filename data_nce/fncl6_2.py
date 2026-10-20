"""
fncl6_2.py — 論文 §6.2「Effect of Intra-layer Correlation」(本文の数値)

各隠れ層のノイズに、層内の全ユニットがサンプルごとに共有するガウス成分を混ぜる:
    eta_k = sigma * ( sqrt(1 - rho) eps_k + sqrt(rho) eps_c ),   rho in [0, 1)
rho が層内のゆらぎ相関を直接制御する (rho = 0 が本論文の通常設定;
nnn.layer.GaussianCrossingSampleLayer.common_mode)。
§5.1 の sin(x) プロトコル上で rho ごとに

  (a) 未学習状態の weight mirror の r (8 パス) と、実際に生じた層内ゆらぎ相関
      (入力内で中心化した z の、ユニット対の相関係数の平均),
  (b) cov_jac の更新方向と backprop 参照の層別 cosine (未学習, 32 draw),
  (c) 最終 MSE: backprop / cov_jac (単変量 mirror = 本論文) / cov_jac (多変量 mirror)

を測る。単変量 mirror のバイアスは W_hat = W R, R_ki = Cov(z_k, z_i)/Var(z_i) であり、
多変量回帰 Cov(d, z) Cov(z, z)^{-1} はこれを除く (§6.2)。

生成物 (out/fncl6_2/):
  results.json   -> 数値一式
  (標準出力)     -> rho ごとの数表 (ゆらぎ相関, mirror r, cosine, 最終 MSE)

実行例:
  python data_nce/fncl6_2.py --device cuda
  python data_nce/fncl6_2.py --quick
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fncl_common import (add_common_args, config_dict, fidelity_untrained, finalize_args,
                         make_task, mean_std, model_factory, run_method, save_json, fncl)
from fncl5_3 import collect_samples

RHOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def set_common_mode(net, rho: float):
    """各隠れ層のガウス交差層に共通モード比 rho を設定する."""
    for cl in net.gaussian_crossing:
        cl.common_mode = rho
    return net


def fluctuation_corr(net, x, passes: int = 4) -> dict:
    """層内ゆらぎ相関: 入力内で中心化した z の、ユニット対の相関係数の平均 (層別)."""
    s = collect_samples(net, x, passes)
    out = {}
    for l, z in enumerate(s["z"]):
        cz = z - z.mean(dim=1, keepdim=True)                       # [N, T, H]
        cov = torch.einsum("ntk,nti->ki", cz, cz) / (cz.shape[0] * cz.shape[1])
        sd = cov.diagonal().clamp_min(1e-12).sqrt()
        corr = cov / (sd[:, None] * sd[None, :])
        H = corr.shape[0]
        off = corr[~torch.eye(H, dtype=torch.bool, device=corr.device)]
        out[f"z{l + 1}"] = float(off.mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="§6.2 intra-layer correlation sweep.")
    add_common_args(p)
    p.add_argument("--rhos", type=str, default=",".join(str(r) for r in RHOS))
    p.add_argument("--grad-draws", type=int, default=32)
    args = finalize_args(p.parse_args(), default_out="out/fncl6_2")
    rhos = [float(r) for r in args.rhos.split(",")]
    if args.quick:
        rhos, args.grad_draws = rhos[:2], 4
    device = torch.device(args.device)
    noise = "gaussian"
    x_raw, target, x, t = make_task(device)

    results = {"config": config_dict(args), "rhos": rhos, "runs": {}}
    for rho in rhos:
        rec = {"final_mse": {"backprop": {}, "cov_jac": {}, "cov_jac_mv": {}},
               "fidelity": {}, "fluct_corr": {}}
        for seed in args.seed_list:
            # backprop (同じ相関ノイズの下)
            torch.manual_seed(seed)
            np.random.seed(seed)
            fresh0 = model_factory(noise, args, device)
            fresh = lambda f=fresh0, r=rho: set_common_mode(f(), r)  # noqa: E731
            _, pred, _ = run_method({"kind": "backprop"}, fresh, x, t, noise, args)
            rec["final_mse"]["backprop"][seed] = float(np.mean((pred - target) ** 2))
            # cov_jac (単変量 mirror) と多変量 mirror
            for name, mirror in (("cov_jac", "intrinsic"), ("cov_jac_mv", "mv")):
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = fresh()
                fncl.train_jac_flex(net, x, t, args.lr, args.epochs, opt="adam",
                                    mirror=mirror, jac_ema=args.jac_ema, eval_every=0)
                pred = fncl.predict(net, x)
                rec["final_mse"][name][seed] = float(np.mean((pred - target) ** 2))
            print(f"[rho={rho}] seed {seed} " + "  ".join(
                f"{k}={v[seed]:.5f}" for k, v in rec["final_mse"].items()), flush=True)
            if seed == args.seed_list[0]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                net = fresh()
                rec["fidelity"] = fidelity_untrained(net, x, t, draws=args.grad_draws)
                rec["fluct_corr"] = fluctuation_corr(net, x)
                print(f"[rho={rho}] fluct corr {rec['fluct_corr']} | "
                      + " ".join(f"cos({k})={c:.3f}" for k, c in rec["fidelity"]["cos"].items())
                      + f" | mirror r w1={rec['fidelity']['mirror_r']['w1']:.4f}"
                      f" wout={rec['fidelity']['mirror_r']['wout']:.4f}", flush=True)
        results["runs"][str(rho)] = rec
        save_json(args.out_dir / "results.json", results)

    lines = ["| rho | fluct. corr (z1 / z2) | mirror r (w1 / wout) | cos (w0 / w1) "
             "| backprop | cov_jac (univariate) | cov_jac (multivariate) |",
             "|---|---|---|---|---|---|---|"]
    for rho in rhos:
        rec = results["runs"][str(rho)]
        fc, fid = rec["fluct_corr"], rec["fidelity"]
        cells = []
        for m in ("backprop", "cov_jac", "cov_jac_mv"):
            mu, sd = mean_std(rec["final_mse"][m].values())
            cells.append(f"{mu:.5f} ± {sd:.5f}")
        lines.append(f"| {rho} | {fc['z1']:.3f} / {fc['z2']:.3f}"
                     f" | {fid['mirror_r']['w1']:.4f} / {fid['mirror_r']['wout']:.4f}"
                     f" | {fid['cos']['w0']:.3f} / {fid['cos']['w1']:.3f} | "
                     + " | ".join(cells) + " |")
    print("\n" + "\n".join(lines) + "\n")   # 表は標準出力のみ (ファイル保存しない)


if __name__ == "__main__":
    main()
