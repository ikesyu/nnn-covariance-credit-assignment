"""
fncl5_2.py — 論文 §5.2「主結果 — cov_jac_adam は backprop 級」(ガウスノイズ)

検証 8 手法 (backprop / cov_only / cov_deriv_analytic / cov_deriv_kde /
cov_jac_sgd / cov_jac_adam / cov_jac_full_sgd / cov_jac_full_adam) を複数 seed で学習し、
以下を生成する:

  (標準出力)                          -> Table 2 (最終 MSE 表, seed 別 + mean±std)
  out/fncl5_2/fig_learning_curves.png -> Fig. 2(a) 相当の確認図 (学習曲線, 先頭 seed)
  out/fncl5_2/fig_fit_check.png       -> Fig. 2(b)(c) 相当の確認図 (fit check: target vs
                                          backprop vs cov_jac_adam + 残差)
  out/fncl5_2/fig_predictions.png     -> 補助図 (全手法の予測)
  out/fncl5_2/curves_preds.npz        -> Fig. 2 の素材 (fncl5_2_fig.py が合成図を作る)
  out/fncl5_2/results.json            -> 数値一式 (config + final MSE)

実行例:
  python data_nce/fncl5_2.py                 # 3 seeds x 1500 epochs (時間がかかる)
  python data_nce/fncl5_2.py --seeds 0,1,2,3,4
  python data_nce/fncl5_2.py --quick         # 動作確認
"""
import argparse

from fncl_common import add_common_args, finalize_args, run_verification


def main() -> None:
    p = argparse.ArgumentParser(
        description="§5.2 main result: cov_jac_adam reaches backprop level "
                    "(gaussian noise).")
    add_common_args(p)
    p.add_argument("--noise", choices=("gaussian", "uniform"), default="gaussian")
    args = finalize_args(p.parse_args(), default_out="out/fncl5_2")
    run_verification(args.noise, args)


if __name__ == "__main__":
    main()
