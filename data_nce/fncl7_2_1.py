"""
fncl7_2_1.py — 論文 §7.2「Noise Selection for Sparseness」の一様ノイズ実験 (Table 7)

§5.2 と同一プロトコル (検証 8 手法 x 複数 seed) を --noise uniform
(SimpleNNNUniformSample, 半幅 r = 1.0) で実行し、主結果の 2 点

  (i)  cov_jac_adam が backprop 水準に達すること,
  (ii) 分布フリー kde slope が解析 phi' (一様: -(d-c)/r^2) と一致すること
       (= 一様ノイズの解析式を使わずに同じ精度)

が一様ノイズでも成り立つことを確認する。§7 のデジタル HW 主張の実験的裏付け。
Table 7 のガウス列は Table 2 (fncl5_2.py) の再掲で、一様列をこのスクリプトが与える。
離散 (b ビット) 一様ノイズの実験 (Fig. 7) は fncl7_2_2.py が担当する。

生成物 (out/fncl7_2_1/):
  (標準出力)              -> Table 7 の uniform 列の最終 MSE 表
  fig_learning_curves.png -> 補助図 (uniform 学習曲線)
  fig_fit_check.png       -> 補助図 (uniform fit check)
  fig_predictions.png
  curves_preds.npz        -> fncl5_2_fig.py --data out/fncl7_2_1 で Fig. 2 と同形の合成図
  results.json

実行例:
  python data_nce/fncl7_2_1.py
  python data_nce/fncl7_2_1.py --radius 1.0 --quick
"""
import argparse

from fncl_common import add_common_args, finalize_args, run_verification


def main() -> None:
    p = argparse.ArgumentParser(
        description="§7.2 uniform-noise reproduction of the main result (Table 7).")
    add_common_args(p)
    args = finalize_args(p.parse_args(), default_out="out/fncl7_2_1")
    run_verification("uniform", args)


if __name__ == "__main__":
    main()
