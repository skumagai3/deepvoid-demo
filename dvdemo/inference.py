"""Full-volume prediction from overlapping patches.

The box is periodic, so patches wrap around its faces. Each patch of side
``size`` is predicted, and only its central ``stride``-wide cube is kept;
the margins (``(size - stride) / 2`` voxels each side) provide context and
are discarded, which removes the seams that edge effects of zero padding
would otherwise leave. With the training subcube size, size=32 and stride=16, a 128^3 box
takes 8^3 = 512 patches and every voxel has 8 voxels of context on all sides.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def tile_plan(n: int, size: int, stride: int) -> List[Tuple[int, int]]:
    """(patch_start, keep_start) pairs along one periodic axis of length n."""
    if (size - stride) % 2:
        raise ValueError("size - stride must be even")
    if n % stride:
        raise ValueError(f"stride {stride} must divide the axis length {n}")
    m = (size - stride) // 2
    return [(k - m, k) for k in range(0, n, stride)]


def _wrap_extract(vol: np.ndarray, start, size: int) -> np.ndarray:
    idx = [(s + np.arange(size)) % n for s, n in zip(start, vol.shape[:3])]
    return vol[np.ix_(*idx)]


def predict_volume(predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                   size: int = 32, stride: int = 16, batch_size: int = 32,
                   num_classes: int = 4, verbose: bool = False) -> np.ndarray:
    """Return class probabilities of shape (X, Y, Z, C) for a normalized input.

    ``predict_fn`` maps a batch (B, size, size, size, 1) to probabilities
    (B, size, size, size, C); pass ``model.predict_on_batch`` for a Keras model.
    """
    m = (size - stride) // 2
    plans = [tile_plan(n, size, stride) for n in x.shape[:3]]
    jobs = [(a, b, c) for a in plans[0] for b in plans[1] for c in plans[2]]
    out = np.zeros(x.shape[:3] + (num_classes,), dtype=np.float32)
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i:i + batch_size]
        batch = np.stack([_wrap_extract(x, [p[0] for p in job], size) for job in chunk])[..., None]
        probs = np.asarray(predict_fn(batch.astype(np.float32)))
        for job, p in zip(chunk, probs):
            (_, ki), (_, kj), (_, kk) = job
            out[ki:ki + stride, kj:kj + stride, kk:kk + stride] = \
                p[m:m + stride, m:m + stride, m:m + stride]
        if verbose:
            print(f"\rpredicted {min(i + batch_size, len(jobs))}/{len(jobs)} patches", end="")
    if verbose:
        print()
    return out


def keras_predict_fn(model):
    """Wrap a Keras model for :func:`predict_volume` (works on CPU or GPU)."""
    def fn(batch):
        return model(batch, training=False).numpy()
    return fn
