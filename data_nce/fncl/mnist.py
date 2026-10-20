"""fncl.mnist — MNIST 部分集合上の多クラス NNN (論文 §6.3, §6.4) の共通部.

- load_mnist / load_split: 固定 seed (12345) の class-balanced 抽出と、
  固定置換 (seed 999) による学習／テスト集合への素な分割 (§6.4 の記述どおり),
- build_net: 784-width-...-width-10 の Sample モデル (ガウス／一様ノイズ),
- collect_samples / ce_error / per_sample_ce / autograd_gradient / averaged:
  忠実度測定 (fncl5_3.py の多クラス交差エントロピー版),
- manual_grads / train_cov_jac_ce / train_backprop_ce / evaluate:
  cov_jac (§4.3: 共分散 weight mirror + 再帰 + KDE slope + 局所 Adam,
  EMA と Kolen-Pollack 追従つき) の交差エントロピー多出力版と backprop 参照.

MNIST は初回に ossci-datasets から学習集合 (idx 形式) をダウンロードし、
`mnist_dir` (既定: カレントディレクトリの data/mnist/) に保存する。
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

from .metrics import cosine, norm_ratio, pearson
from .network import Capture, crossing_layers, kde_slope
from .train import ManualOpt, cov_weight

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DEFAULT_MNIST_DIR = Path("data") / "mnist"
SUBSET_SEED = 12345        # class-balanced 抽出の seed
SPLIT_SEED = 999           # 学習／テストへの分割置換の seed


# ============================================================
# MNIST (raw idx, 依存ライブラリなし)
# ============================================================
def _fetch(name: str, mnist_dir: Path) -> Path:
    mnist_dir.mkdir(parents=True, exist_ok=True)
    path = mnist_dir / name
    if not path.exists():
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(MNIST_URL + name, path)
    return path


def load_mnist(n: int, seed: int = SUBSET_SEED, mnist_dir=None):
    """MNIST 学習集合から class-balanced に n 枚を選ぶ。x in [-1,1]^784, y in {0..9}."""
    mnist_dir = Path(mnist_dir) if mnist_dir is not None else DEFAULT_MNIST_DIR
    with gzip.open(_fetch("train-images-idx3-ubyte.gz", mnist_dir)) as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(_fetch("train-labels-idx1-ubyte.gz", mnist_dir)) as f:
        _ = f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    idx = []
    per_class = n // 10
    for c in range(10):
        cand = np.where(labels == c)[0]
        idx.append(rng.choice(cand, per_class, replace=False))
    idx = rng.permutation(np.concatenate(idx))
    x = images[idx].astype(np.float32) / 255.0 * 2.0 - 1.0
    y = labels[idx].astype(np.int64)
    return x, y


def load_split(n_inputs: int, n_test: int, device, mnist_dir=None):
    """n_inputs + n_test 枚を 1 回だけ抽出し、固定置換で学習／テストに素に分割する.

    返り値: (x [N,784], labels [N], (x_test, y_test))。n_test == 0 なら test は (None, None)。
    """
    x_all, y_all = load_mnist(n_inputs + n_test, mnist_dir=mnist_dir)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(y_all))
    tr, te = perm[:n_inputs], perm[n_inputs:]
    x = torch.tensor(x_all[tr], device=device)
    labels = torch.tensor(y_all[tr], device=device)
    if n_test > 0:
        test = (torch.tensor(x_all[te], device=device),
                torch.tensor(y_all[te], device=device))
    else:
        test = (None, None)
    return x, labels, test


# ============================================================
# モデル・サンプル収集 (fncl5_3 の任意深さ／多クラス CE 一般化)
# ============================================================
def build_net(depth: int, width: int, scale: float, h: float, t: int, device,
              noise: str = "gaussian"):
    """784-width*depth-10 の NNN。scale はガウスなら std、一様なら半幅 radius."""
    from nnn import model as nnn_model
    structure = [784] + [width] * depth + [10]
    if noise == "gaussian":
        net = nnn_model.SimpleNNNSample(structure=structure, std=scale, h=h, t=t,
                                        output_bias=True)
    elif noise == "uniform":
        net = nnn_model.SimpleNNNUniformSample(structure=structure, radius=scale,
                                               center=0.0, h=h, t=t, output_bias=True)
    else:
        raise ValueError(noise)
    return net.to(device)


def collect_samples(net, x, passes: int = 1) -> dict:
    """forward フックで per-sample の d, z, 読み出しサンプル ys を passes 回分集める."""
    cap = Capture(net)
    z = [[] for _ in range(cap.n_hidden)]
    d = [[] for _ in range(cap.n_hidden)]
    ys, y_out = [], []
    with torch.no_grad():
        for _ in range(passes):
            y_out.append(net(x))
            ys.append(cap.y_samples)
            for l in range(cap.n_hidden):
                z[l].append(cap.z[l])
                d[l].append(cap.d[l])
    cap.remove()
    return {"y": torch.stack(y_out, dim=0).mean(dim=0),      # [N, 10]
            "ys": torch.cat(ys, dim=1),                      # [N, K*T, 10]
            "z": [torch.cat(zl, dim=1) for zl in z],
            "d": [torch.cat(dl, dim=1) for dl in d]}


def ce_error(y_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """アンサンブル平均 logit での交差エントロピーの dL/dy = softmax - onehot."""
    return torch.softmax(y_logits, dim=1) - tF.one_hot(labels, 10).float()


def per_sample_ce(ys: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """各確率サンプルの logit y_(m) に対する per-sample 交差エントロピー [N, T]."""
    N, T, C = ys.shape
    return tF.cross_entropy(ys.reshape(N * T, C),
                            labels.unsqueeze(1).expand(N, T).reshape(N * T),
                            reduction="none").view(N, T)


def autograd_gradient(net, x, labels) -> dict:
    """backprop 参照 (同じ確率的モデル上の 1 draw の autograd 勾配)."""
    for prm in net.parameters():
        prm.grad = None
    loss = tF.cross_entropy(net(x), labels)
    loss.backward()
    grads = {f"w{l}": net.fcs[l].weight.grad.detach().clone()
             for l in range(len(net.fcs) - 1)}
    grads["wout"] = net.fcs[-1].weight.grad.detach().clone()
    return grads


def averaged(fn, draws: int) -> dict:
    acc = None
    for _ in range(draws):
        g = fn()
        acc = g if acc is None else {k: acc[k] + v for k, v in g.items()}
    return {k: v / draws for k, v in acc.items()}


def pretrain_backprop(net, x, labels, lr: float, epochs: int, log_every: int = 0):
    """backprop (Adam) による事前学習 (§6.3 の忠実度測定の「学習後」状態を作る)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        logits = net(x)
        loss = tF.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            acc = float((logits.argmax(1) == labels).float().mean())
            print(f"  [pretrain] epoch {epoch:4d}  CE={float(loss):.4f}  "
                  f"acc={acc:.3f}", flush=True)


# ============================================================
# cov_jac の交差エントロピー多出力版 (§6.4)
# ============================================================
def evaluate(net, x, labels, x_te=None, y_te=None):
    """(train CE, train acc, test acc) を返す (テスト集合が無ければ test acc は nan)."""
    with torch.no_grad():
        logits = net(x)
        ce = float(tF.cross_entropy(logits, labels))
        acc = float((logits.argmax(1) == labels).float().mean())
        te = float("nan")
        if x_te is not None:
            te = float((net(x_te).argmax(1) == y_te).float().mean())
    return ce, acc, te


def manual_grads(net, x, labels, w_mirror: dict, s: dict, e: torch.Tensor = None):
    """weight mirror w_mirror と収集済みサンプル s から cov_jac の勾配を作る.

    e は読み出し誤差 dL/dy [N, 10]。None なら解析形 (softmax - onehot) を使う。
    w_mirror のキーは "out" (読み出し) と 1..n_hidden-1 (隠れ層間)。
    """
    crossings = crossing_layers(net)
    n_hidden = len(crossings)
    slope_full = [kde_slope(crossings[l], s["d"][l]) for l in range(n_hidden)]
    slope_mean = [sf.mean(dim=1) for sf in slope_full]
    N, T = x.shape[0], s["z"][0].shape[1]
    if e is None:
        e = ce_error(s["y"], labels)                               # [N, 10]
    a = [None] * n_hidden
    a[-1] = e @ w_mirror["out"]
    for l in range(n_hidden - 2, -1, -1):
        a[l] = (a[l + 1] * slope_mean[l + 1]) @ w_mirror[l + 1]
    z_prev = ([x.unsqueeze(1).expand(N, T, x.shape[1])]
              + [s["z"][l] for l in range(n_hidden - 1)])
    gW, gb = {}, {}
    for l in range(n_hidden):
        delta = a[l].unsqueeze(1) * slope_full[l]
        gW[f"w{l}"] = torch.einsum("nto,nti->oi", delta, z_prev[l]) / (N * T)
        gb[f"w{l}"] = delta.mean(dim=(0, 1))
    gW["wout"] = torch.einsum("no,ni->oi", e, s["z"][-1].mean(dim=1)) / N
    gb["wout"] = e.mean(dim=0)
    return gW, gb


def measure_mirrors(s: dict, n_hidden: int) -> dict:
    """収集済みサンプルから全 weight mirror を単変量回帰 (入力プール) で測る."""
    meas = {"out": cov_weight(s["ys"], s["z"][-1], pool=True)}
    for l in range(1, n_hidden):
        meas[l] = cov_weight(s["d"][l], s["z"][l - 1], pool=True)
    return meas


def apply_updates(net, optim: ManualOpt, gW: dict, gb: dict, lr: float,
                  n_hidden: int) -> dict:
    """手動オプティマイザで全層を更新し、重みの変化量 (Kolen-Pollack 追従用) を返す."""
    steps = {}
    for l in range(n_hidden):
        steps[l] = optim.update(f"w{l}", net.fcs[l].weight, gW[f"w{l}"], lr)
        if net.fcs[l].bias is not None:
            optim.update(f"b{l}", net.fcs[l].bias, gb[f"w{l}"], lr)
    steps["out"] = optim.update("wout", net.fcs[-1].weight, gW["wout"], lr)
    if net.fcs[-1].bias is not None:
        optim.update("bout", net.fcs[-1].bias, gb["wout"], lr)
    return steps


def mirror_pearson(net, W_ema: dict, n_hidden: int) -> dict:
    """学習時 weight mirror と真の重みの Pearson r (キー w1.., wout)."""
    mr = {f"w{l}": pearson(W_ema[l], net.fcs[l].weight) for l in range(1, n_hidden)}
    mr["wout"] = pearson(W_ema["out"], net.fcs[-1].weight)
    return mr


def fidelity_at(net, x, labels, W_ema: dict, grad_draws: int) -> dict:
    """学習時 mirror で作る更新方向と backprop 参照 (autograd) の層別 cosine・ノルム比."""
    g_auto = averaged(lambda: autograd_gradient(net, x, labels), grad_draws)

    def one_draw():
        with torch.no_grad():
            ss = collect_samples(net, x, passes=1)
            return manual_grads(net, x, labels, W_ema, ss)[0]

    g_est = averaged(one_draw, grad_draws)
    return {k: {"cos": cosine(g_est[k], g_auto[k]),
                "ratio": norm_ratio(g_est[k], g_auto[k])} for k in g_auto}


def train_cov_jac_ce(net, x, labels, lr: float, epochs: int, jac_ema: float,
                     eval_every: int, grad_draws: int, log=print, test=(None, None)):
    """cov_jac (§4.3) の交差エントロピー多出力版。EMA + Kolen-Pollack 追従つき mirror、局所 Adam.

    eval_every ごとに (train CE, train acc, test acc, mirror r, 層別 cosine) を記録する。
    """
    n_hidden = len(crossing_layers(net))
    optim = ManualOpt("adam")
    W_ema, history = {}, []
    for epoch in range(epochs):
        with torch.no_grad():
            s = collect_samples(net, x, passes=1)
            meas = measure_mirrors(s, n_hidden)              # 今 epoch の共分散 mirror
            if not W_ema:
                W_ema.update(meas)
            else:
                for k, v in meas.items():
                    W_ema[k] = jac_ema * W_ema[k] + (1.0 - jac_ema) * v
            gW, gb = manual_grads(net, x, labels, W_ema, s)
            steps = apply_updates(net, optim, gW, gb, lr, n_hidden)
            # Kolen-Pollack 追従: 既知の更新量を mirror へ直接加算
            W_ema["out"] = W_ema["out"] - steps["out"]
            for l in range(1, n_hidden):
                W_ema[l] = W_ema[l] - steps[l]
        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = evaluate(net, x, labels, *test)
            mirror_r = mirror_pearson(net, W_ema, n_hidden)
            fid = fidelity_at(net, x, labels, W_ema, grad_draws)
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te_acc,
                            "mirror_r": mirror_r, "fidelity": fid})
            log(f"  [cov_jac ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} test={te_acc:.3f}"
                " | mirror " + " ".join(f"{k}={v:.3f}" for k, v in mirror_r.items())
                + " | cos " + " ".join(f"{k}={v['cos']:.3f}" for k, v in fid.items()))
    return history


def train_backprop_ce(net, x, labels, lr: float, epochs: int, eval_every: int,
                      log=print, test=(None, None)):
    """backprop 参照 (autograd + Adam) の交差エントロピー版."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = tF.cross_entropy(net(x), labels)
        loss.backward()
        opt.step()
        if eval_every and (epoch % eval_every == 0 or epoch == epochs - 1):
            ce, acc, te_acc = evaluate(net, x, labels, *test)
            history.append({"epoch": epoch, "ce": ce, "acc": acc, "test_acc": te_acc})
            log(f"  [backprop ep {epoch:4d}] CE={ce:.4f} acc={acc:.3f} test={te_acc:.3f}")
    return history
