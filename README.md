# Reconstructing Backpropagation from Forward Fluctuations in Noise-modulated Neural Networks

Code to reproduce the results of the paper:

> Shuhei Ikemoto, "Reconstructing Backpropagation from Forward Fluctuations in
> Noise-modulated Neural Networks."

This repository contains the complete set of experiment scripts that reproduce
the paper's tables and figures. Run every script from the repository root
(the scripts assume they sit next to the `nnn/` package):

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch / NumPy / Matplotlib

python data_nce/fncl5_2.py               # full run (3 seeds × 1500 epochs)
python data_nce/fncl5_2.py --quick       # reduced smoke-test configuration
```

Figures and numerical records (`results.json` / `*.npz`) are saved to
`out/fncl<sec>_<n>/` in the current directory by default (change with
`--out`). **The tables (corresponding to Tables 2–5) are printed to standard
output only** and are not written to files. The `*_fig.py` scripts regenerate
the figures from the saved records (`results.json` / `*.npz`) of the
corresponding main script, without re-training.

## Correspondence between scripts and the paper

| Script | Paper section | Output |
|---|---|---|
| `fncl5_2.py` (+`fncl5_2_fig.py`) | §5.2 main results (Gaussian noise) | Table 2, Fig. 2 |
| `fncl5_3.py` (+`fncl5_3_fig.py`) | §5.3 fidelity of the gradient reconstruction | Fig. 3 |
| `fncl5_4.py` (+`fncl5_4_fig.py`) | §5.4 ablation study | Fig. 4 |
| `fncl5_5.py` (+`fncl5_5_fig.py`) | §5.5 covariance estimation of the readout error / skewness bias | Fig. 5 |
| `fncl5_6.py` | §5.6 benchmarks (Friedman #1 / Two Moons / Concentric Circles) | Table 3 |
| `fncl6_1.py` | §6.1 hardware resource estimates (standard output only) | Table 4 |
| `fncl6_2.py` (+`fncl5_2_fig.py --data out/fncl6_2`) | §6.2 reproduction under uniform noise | Table 5 |

## Internal structure

- `nnn/` — the Noise-modulated Neural Network (NNN) library: crossing
  activation functions, noise/sampling/readout layers, feedforward models,
  noise samplers and distribution functions, and noise-field utilities.
- `data_nce/fncl/` — the implementation of all learning rules
  (backprop / cov_only / cov_deriv / cov_jac / cov_jac_full). Depends on the
  `nnn/` package (resolved by adding the repository root to `sys.path`).
  - `constants.py` — shared numerical constants (EPS, NUM_POINTS, ...)
  - `network.py` — model construction (`build_model`), local derivatives
    (`phi_prime` / `kde_slope`), forward-hook instrumentation (`Capture`)
  - `perturb.py` — external perturbation gates (`Perturber` etc.; for the
    node-perturbation control experiments)
  - `train.py` — the learning rules (`train_cov`, `covariance_credit`,
    `cov_weight`, `ManualOpt`), the backprop reference (`train_backprop`),
    and inference (`predict`)
  - `viz.py` — diagnostic plots
- `data_nce/fncl_common.py` — the thin shared basis of every
  `fncl5_*.py` / `fncl6_*.py` script: running the (seed × method) experiment
  matrix and saving tables, JSON, and figures.

torch / numpy are re-seeded immediately before each (seed, method) pair, so
all methods under the same seed start from identical initial weights and an
identical random stream. The numbers in the paper were obtained with
seeds 0, 1, 2, H = 64, T = 64, and 1500 epochs (the default values of each
script).

<!--
## Citation

If you use this code, please cite the paper:

```bibtex
@article{ikemoto_nnn_covariance_credit,
  author  = {Shuhei Ikemoto},
  title   = {Reconstructing Backpropagation from Forward Fluctuations in
             Noise-modulated Neural Networks},
  journal = {},
  year    = {},
  note    = {under review}
}
```
<!-->

## License

This project is released under the [MIT License](LICENSE).
