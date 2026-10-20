"""fncl.metrics — 重み・勾配の比較に使う小さな指標 (Pearson r / cosine / ノルム比).

fncl_common.py の同名関数と同じ定義。fncl パッケージ内のモジュール
(mnist / noisefield / mirror) が fncl_common に依存しないようにここに置く。
"""
from __future__ import annotations

import numpy as np
import torch


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().numpy().ravel()
    b = b.detach().cpu().numpy().ravel()
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().ravel()
    b = b.detach().ravel()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.detach().norm() / (b.detach().norm() + 1e-12))
