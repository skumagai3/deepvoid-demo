"""Reading the ``.fvol`` volumes used by DeepVoid, and loading the demo sample.

``.fvol`` is a 256-byte header followed by a raw, C-ordered, little-endian
payload:

    offset  type       field
    0       int32      datatype (unreliable in legacy files, so dtype is passed explicitly)
    12      int32x3    shape (x, y, z)
    24      float32x3  offset
    36      float32x3  box
    256     payload
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

HEADER_BYTES = 256
CLASS_NAMES = ("void", "wall", "filament", "halo")


def read_fvol(path: str, dtype="float32") -> np.ndarray:
    """Read an ``.fvol`` file into an array of shape (X, Y, Z)."""
    dtype = np.dtype(dtype).newbyteorder("<")
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
    shape = tuple(int(v) for v in np.frombuffer(header[12:24], dtype="<i4"))
    n = int(np.prod(shape))
    expected = HEADER_BYTES + n * dtype.itemsize
    size = os.path.getsize(path)
    if size != expected:
        raise ValueError(f"{path}: {size} bytes, but shape {shape} x {dtype.name} implies {expected}")
    data = np.fromfile(path, dtype=dtype, offset=HEADER_BYTES, count=n)
    return data.reshape(shape).astype(dtype.newbyteorder("="))


def write_fvol(path: str, vol: np.ndarray) -> None:
    """Write a float32 ``.fvol`` (used only by the tests)."""
    vol = np.asarray(vol, dtype="<f4")
    h_int = np.array([8, 0, vol.shape[0], *vol.shape], dtype="<i4")
    h_flt = np.array([0, 0, 0, 1, 1, 1], dtype="<f4")
    pad = np.zeros(HEADER_BYTES - h_int.nbytes - h_flt.nbytes, dtype=np.uint8)
    with open(path, "wb") as f:
        for part in (h_int, h_flt, pad, np.ascontiguousarray(vol)):
            part.tofile(f)


def read_mask(path: str) -> np.ndarray:
    """Read a T-web mask stored as float32 and return uint8 class IDs 0..3."""
    raw = read_fvol(path, "float32")
    labels = np.rint(raw)
    if not np.allclose(raw, labels):
        raise ValueError(f"{path}: mask contains non-integer values")
    if labels.min() < 0 or labels.max() > 3:
        raise ValueError(f"{path}: mask values outside 0..3")
    return labels.astype(np.uint8)


def load_sample(path: str) -> Dict[str, object]:
    """Load the committed demo sample (``data/tng300_128.npz``).

    Returns a dict with float32 input fields (``dm`` and ``subhalo_L7``), the
    uint8 ``mask``, and ``meta`` (a dict describing the box, labels and split).
    """
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    for k in meta["fields"]:
        out[k] = out[k].astype(np.float32)
    out["meta"] = meta
    return out
