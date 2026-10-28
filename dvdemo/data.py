"""Subcube split, normalization and training data.

* the 128^3 box is cut into 32^3 subcubes (4 x 4 x 4 = 64 of them);
* subcubes are split at random into training and validation (80 / 20,
  seed 12), *before* any augmentation;
* only training subcubes are augmented, with four copies each: as-is and
  rotated 90 degrees about each of the three axes;
* validation subcubes are never rotated or trained on, and all metrics
  are computed on their voxels;
* input scaling (``Normalizer``) is a choice: min-max, or log10(1 + delta)
  with min-max or z-score scaling.

With ``overlap=True`` training subcubes are also taken at a stride of 16
voxels, keeping only those lying entirely inside training subcubes, so no
training input ever contains validation labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

SUBGRID = 32        # subcube side for the 128^3 box
OFF = 16            # stride of the optional overlapping training subcubes
VAL_FRACTION = 0.2
SEED = 12

Cell = Tuple[int, int, int]


def make_split(n: int = 128, subgrid: int = SUBGRID, val_fraction: float = VAL_FRACTION,
               seed: int = SEED) -> Dict[str, object]:
    """Assign each subcube (cell) of the box to training or validation."""
    if n % subgrid:
        raise ValueError(f"subgrid {subgrid} must divide the box size {n}")
    m = n // subgrid
    cells = [(i, j, k) for i in range(m) for j in range(m) for k in range(m)]
    order = np.random.RandomState(seed).permutation(len(cells))
    n_val = int(np.ceil(val_fraction * len(cells)))
    val = sorted(cells[i] for i in order[:n_val])
    train = sorted(cells[i] for i in order[n_val:])
    return {"subgrid": subgrid, "train": [list(c) for c in train],
            "val": [list(c) for c in val], "seed": seed}


def cell_mask(shape: Sequence[int], cells: Sequence[Sequence[int]], subgrid: int) -> np.ndarray:
    """Boolean mask of all voxels inside the listed subcubes."""
    mask = np.zeros(tuple(shape[:3]), dtype=bool)
    for i, j, k in cells:
        mask[i * subgrid:(i + 1) * subgrid, j * subgrid:(j + 1) * subgrid,
             k * subgrid:(k + 1) * subgrid] = True
    return mask


def split_masks(shape, split: Dict) -> Tuple[np.ndarray, np.ndarray]:
    s = split["subgrid"]
    return cell_mask(shape, split["train"], s), cell_mask(shape, split["val"], s)


def check_split(split: Dict, n: int = 128) -> None:
    """Raise unless training and validation subcubes are disjoint and tile the box."""
    train, val = ({tuple(c) for c in split[k]} for k in ("train", "val"))
    if train & val:
        raise ValueError(f"subcubes in both training and validation: {sorted(train & val)}")
    if len(train | val) != (n // split["subgrid"]) ** 3:
        raise ValueError("split does not cover every subcube exactly once")


LOG_FLOOR = 1e-3    # floor on rho/rho_bar before the log (empty voxels of sparse fields)
METHODS = ("minmax", "log", "log_zscore")


@dataclass
class Normalizer:
    """Input scaling, fitted on the training voxels and saved with the model.

    The stored fields are rho / rho_bar = 1 + delta. Three choices:

    ``minmax``      (x - min) / (max - min). The paper's choice: simple,
                    bounded to [0, 1], no free parameters.
    ``log``         log10(1 + delta) = log10(rho / rho_bar), floored at
                    LOG_FLOOR, then min-max to [0, 1]. Density fields are close
                    to log-normal, so this spreads out the low-density range
                    where voids and walls differ.
    ``log_zscore``  the same log, then (y - mean) / std. Depends on the bulk of
                    the distribution rather than its single most extreme voxel.
    """
    method: str = "minmax"
    stats: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown normalization {self.method!r}; choose from {METHODS}")

    def _pre(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.method == "minmax":
            return x
        return np.log10(np.maximum(x, LOG_FLOOR))

    def fit(self, x: np.ndarray) -> "Normalizer":
        y = self._pre(x)
        if self.method == "log_zscore":
            self.stats = {"mean": float(y.mean()), "std": float(y.std())}
            ok = self.stats["std"] > 0
        else:
            self.stats = {"min": float(y.min()), "max": float(y.max())}
            ok = self.stats["max"] > self.stats["min"]
        if not ok:
            raise ValueError("cannot normalize a constant field")
        if self.method != "minmax":
            self.stats["floor"] = LOG_FLOOR
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.stats:
            raise RuntimeError("Normalizer.transform called before fit")
        y, st = self._pre(x), self.stats
        if self.method == "log_zscore":
            out = (y - st["mean"]) / st["std"]
        else:
            out = (y - st["min"]) / (st["max"] - st["min"])
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(method=d.get("method", "minmax"), stats=dict(d["stats"]))


def rotations(cube: np.ndarray) -> List[np.ndarray]:
    """The four training copies of a subcube: as-is, and
    rotated 90 degrees about each of the three axes."""
    return [cube, np.rot90(cube, axes=(0, 1)), np.rot90(cube, axes=(0, 2)),
            np.rot90(cube, axes=(1, 2))]


def _cut(vol, corner, s):
    i, j, k = corner
    return vol[i:i + s, j:j + s, k:k + s]


def training_corners(split: Dict, n: int = 128, overlap: bool = False) -> List[Cell]:
    """Corners of training subcubes. With ``overlap``, also subcubes at a stride
    of OFF that lie entirely inside training cells."""
    s = split["subgrid"]
    if not overlap:
        return [tuple(c * s for c in cell) for cell in split["train"]]
    train = {tuple(c) for c in split["train"]}
    starts = range(0, n - s + 1, OFF)
    out = []
    for a in starts:
        for b in starts:
            for c in starts:
                touched = {(i // s, j // s, k // s) for i in (a, a + s - 1)
                           for j in (b, b + s - 1) for k in (c, c + s - 1)}
                if touched <= train:
                    out.append((a, b, c))
    return out


def build_arrays(x_norm: np.ndarray, y: np.ndarray, split: Dict, overlap: bool = False,
                 augment: bool = True):
    """Training arrays (augmented) and validation arrays (not augmented),
    shaped (N, s, s, s, 1) for Keras."""
    s = split["subgrid"]
    xs, ys = [], []
    for corner in training_corners(split, x_norm.shape[0], overlap):
        xc, yc = _cut(x_norm, corner, s), _cut(y, corner, s)
        pairs = zip(rotations(xc), rotations(yc)) if augment else [(xc, yc)]
        for a, b in pairs:
            xs.append(a); ys.append(b)
    xv = [_cut(x_norm, [c * s for c in cell], s) for cell in split["val"]]
    yv = [_cut(y, [c * s for c in cell], s) for cell in split["val"]]

    def stack(v, dtype):
        return np.ascontiguousarray(np.stack(v)[..., None], dtype=dtype)

    return (stack(xs, np.float32), stack(ys, np.int32),
            stack(xv, np.float32), stack(yv, np.int32))


def val_cells_in_slice(split: Dict, index: int, axis: int = 0) -> List[Cell]:
    """Validation subcubes that a slice at ``index`` along ``axis`` passes through."""
    s = split["subgrid"]
    return [tuple(c) for c in split["val"] if c[axis] == index // s]
