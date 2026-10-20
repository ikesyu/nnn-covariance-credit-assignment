"""fncl — Forward-Noise Covariance Learning for Noise-modulated Neural Networks.

論文「Reconstructing Backpropagation from Forward Fluctuations in Noise-modulated
Neural Networks」の学習則 (backprop / cov_only / cov_deriv / cov_jac /
cov_jac_full と gate 変種) の実装本体。旧 forward_noise_covariance_learning.py
(モノリス) を役割別に分割したもの:

  constants.py : 共有数値定数 (EPS, NUM_POINTS, ...)
  network.py   : モデル構築 (build_model), 局所微分 (phi_prime / kde_slope),
                 forward フック計測 (Capture)
  perturb.py   : 外部摂動ゲート (Perturber, gate_masks, rng snapshot; Appendix B)
  train.py     : 学習則 (train_cov, covariance_credit, cov_weight, ManualOpt) と
                 backprop 参照 (train_backprop), 推論 (predict)
  viz.py       : 確認用プロット
  metrics.py   : 比較指標 (pearson / cosine / norm_ratio)
  mirror.py    : weight mirror の推定法を差し替えられる cov_jac ループ (§6.2, §6.3)
  baselines.py : 代替 forward-only 学習則 FA / DFA / PEPITA (§6.3)
  mnist.py     : MNIST 部分集合と交差エントロピー多出力版の cov_jac (§6.3, §6.4)
  noisefield.py: サンプル軸に周期構造を持つノイズ場と位相内中心化 (§6.4)

アルゴリズムの詳細は docs/forward_noise_covariance_learning.md を参照。
"""
from .constants import EPS, NUM_POINTS, SQRT2, UNIFORM_CENTER
from .network import (Capture, build_model, crossing_layers, gauss_cdf,
                      gauss_pdf, kde_slope, phi_prime)
from .perturb import Perturber, gate_masks, rng_restore, rng_snapshot
from .train import (ManualOpt, cov_weight, covariance_credit, lr_at, predict,
                    train_backprop, train_cov)
from .viz import (plot_activity_stats, plot_fit_check, plot_losses,
                  plot_predictions)
from .metrics import cosine, norm_ratio, pearson
from .mirror import mv_mirror, probe_mirror, train_jac_flex
from .baselines import grads_from, random_feedback, train_fa, train_pepita
from .mnist import (apply_updates, autograd_gradient, averaged, build_net, ce_error,
                    collect_samples, evaluate, fidelity_at, load_mnist, load_split,
                    manual_grads, measure_mirrors, mirror_pearson, per_sample_ce,
                    pretrain_backprop, train_backprop_ce, train_cov_jac_ce)
from .noisefield import (PeriodicGaussianNoise, PeriodicUniformCrossing,
                         cov_weight_phase, install_noise_field, phase_center,
                         restore_noise_field, train_cov_jac_ce_noisefield)

__all__ = [
    "EPS", "NUM_POINTS", "SQRT2", "UNIFORM_CENTER",
    "Capture", "build_model", "crossing_layers", "gauss_cdf", "gauss_pdf",
    "kde_slope", "phi_prime",
    "Perturber", "gate_masks", "rng_restore", "rng_snapshot",
    "ManualOpt", "cov_weight", "covariance_credit", "lr_at", "predict",
    "train_backprop", "train_cov",
    "plot_activity_stats", "plot_fit_check", "plot_losses", "plot_predictions",
    "cosine", "norm_ratio", "pearson",
    "mv_mirror", "probe_mirror", "train_jac_flex",
    "grads_from", "random_feedback", "train_fa", "train_pepita",
    "apply_updates", "autograd_gradient", "averaged", "build_net", "ce_error",
    "collect_samples", "evaluate", "fidelity_at", "load_mnist", "load_split",
    "manual_grads", "measure_mirrors", "mirror_pearson", "per_sample_ce",
    "pretrain_backprop", "train_backprop_ce", "train_cov_jac_ce",
    "PeriodicGaussianNoise", "PeriodicUniformCrossing", "cov_weight_phase",
    "install_noise_field", "phase_center", "restore_noise_field",
    "train_cov_jac_ce_noisefield",
]
