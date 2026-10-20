# Reconstructing Backpropagation from Forward Fluctuations in Noise-modulated Neural Networks

Code to reproduce the results of the paper:

> Shuhei Ikemoto, "Reconstructing Backpropagation from Forward Fluctuations in
> Noise-modulated Neural Networks."

This repository contains the complete set of experiment scripts that reproduce
the paper's tables and figures, including the extended evaluation of Section 6
and the hardware-oriented analysis of Section 7. Run every script from the
repository root (the scripts assume they sit next to the `nnn/` package):

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch / NumPy / Matplotlib

python data_nce/fncl5_2.py               # full run (3 seeds x 1500 epochs)
python data_nce/fncl5_2.py --quick       # reduced smoke-test configuration
python data_nce/fncl6_4_1.py --device cuda   # use a GPU where available
```

Figures and numerical records (`results.json` / `*.npz`) are saved to
`out/fncl<section>_<subsection>[_<n>]/` in the current directory by default
(change with `--out`). **The tables are printed to standard output only** and
are not written to files. The `*_fig.py` scripts regenerate the figures from
the saved records of the corresponding main script, without re-training.
Scripts are named `fncl<section>_<subsection>.py`; when a subsection has more
than one experiment, the scripts are numbered `fncl<section>_<subsection>_<n>.py`.

## Correspondence between scripts and the paper

| Script | Paper section | Output |
|---|---|---|
| `fncl5_2.py` (+`fncl5_2_fig.py`) | §5.2 main results (Gaussian noise) | Table 2, Fig. 2 |
| `fncl5_3.py` (+`fncl5_3_fig.py`) | §5.3 accuracy of the gradient reconstruction | Fig. 3 |
| `fncl5_4.py` (+`fncl5_4_fig.py`) | §5.4 ablation study | Fig. 4 |
| `fncl5_5.py` (+`fncl5_5_fig.py`) | §5.5 covariance estimation of the readout error / skewness bias | Fig. 5 |
| `fncl5_6.py` | §5.6 benchmarks (Friedman #1 / Two Moons / Concentric Circles) | Table 3 |
| `fncl6_1.py` (+`fncl6_1_fig.py`) | §6.1 sensitivity to T, noise intensity, h, and ε | Fig. 6 |
| `fncl6_2.py` | §6.2 effect of intra-layer correlation | numbers in the text |
| `fncl6_3_1.py` | §6.3 alternative forward-only rules (FA / DFA / node perturbation / PEPITA) | Table 4, MSE columns |
| `fncl6_3_2.py` | §6.3 update-direction cosine on the MNIST network | Table 4, last column |
| `fncl6_3_3.py` | §6.3 dedicated-phase weight-mirror baseline | numbers in the text |
| `fncl6_4_1.py` | §6.4 depth dependence and the noise field on MNIST | Table 5(a) |
| `fncl6_4_2.py` | §6.4 readout-error estimators under the cross-entropy loss | Table 5(b) |
| `fncl7_1.py` | §7.1 operation and memory counts (standard output only) | Table 6 |
| `fncl7_2_1.py` (+`fncl5_2_fig.py --data out/fncl7_2_1`) | §7.2 reproduction under uniform noise | Table 7 |
| `fncl7_2_2.py` (+`fncl7_2_2_fig.py`) | §7.2 digitally generated (b-bit / LFSR) uniform noise | Fig. 7 |

The backprop and `cov_jac` rows of Table 4 are those of Tables 2 and 3. The
Gaussian column of Table 7 is that of Table 2.

## MNIST subsets

`fncl6_3_2.py`, `fncl6_4_1.py`, and `fncl6_4_2.py` download the MNIST training
set (idx format) on first use into `data/mnist/` (change with `--mnist-dir`).
`fncl6_4_1.py` and `fncl6_4_2.py` draw a class-balanced subset of 2000 images
with a fixed seed (12345) and split it with a fixed permutation (seed 999) into
1000 training and 1000 test images (the two halves are not exactly balanced), so
the same images are used for every learning seed, as described in Section 6.4.
`fncl6_3_2.py` draws a separate class-balanced subset of 250 images
(`--n-inputs 256`, 25 per class) and uses it for both pretraining and measurement.

## Internal structure

- `nnn/` — the Noise-modulated Neural Network (NNN) library: crossing
  activation functions, noise/sampling/readout layers, feedforward models,
  noise samplers and distribution functions, and noise-field utilities.
  `layer.py` and `noise.py` also provide the common-mode noise of Section 6.2
  and the b-bit / LFSR uniform noise of Section 7.2 (off by default).
- `data_nce/fncl/` — the implementation of all learning rules. Depends on the
  `nnn/` package (resolved by adding the repository root to `sys.path`).
  - `constants.py` — shared numerical constants (EPS, NUM_POINTS, ...)
  - `network.py` — model construction (`build_model`), local derivatives
    (`phi_prime` / `kde_slope`), forward-hook instrumentation (`Capture`)
  - `perturb.py` — external perturbation gates (`Perturber` etc.; node perturbation)
  - `train.py` — the learning rules (`train_cov`, `covariance_credit`,
    `cov_weight`, `ManualOpt`), the backprop reference (`train_backprop`),
    and inference (`predict`)
  - `viz.py` — diagnostic plots
  - `metrics.py` — Pearson correlation, cosine similarity, norm ratio
  - `mirror.py` — `cov_jac` with interchangeable weight-mirror estimators
    (intrinsic fluctuations, multivariate regression, dedicated perturbation phase)
  - `baselines.py` — feedback alignment, direct feedback alignment, PEPITA,
    and the update directions compared in Table 4
  - `mnist.py` — MNIST subsets, the multiclass NNN, and the cross-entropy
    versions of `cov_jac` and backprop
  - `noisefield.py` — the sample-axis noise field (periodic upstream noise,
    phase-class centering, rotation of the mirror measurement)
- `data_nce/fncl_common.py` — the thin shared basis of every `fncl*_*.py`
  script: running the (seed × method) experiment matrix, saving tables, JSON,
  and figures, and measuring the gradient fidelity in the untrained network.

torch / numpy are re-seeded immediately before each (seed, method) pair, so
all methods under the same seed start from identical initial weights and an
identical random stream. The numbers in the paper were obtained with
seeds 0, 1, 2 and the default values of each script (H = 64, T = 64, and
1500 epochs on the sin(x) task; 256 units, T = 64, and 400 epochs on MNIST).
The exception is `fncl5_3.py` (Fig. 3, Section 5.3), whose defaults are H = 32 and seed 0.
`kde_slope` re-runs the layer with independent noise to obtain phi'_T (it does not reuse
the forward-pass samples), so the reproduction code draws more random numbers than the
count in Table 6 of the paper, which assumes reuse of the forward samples.

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
