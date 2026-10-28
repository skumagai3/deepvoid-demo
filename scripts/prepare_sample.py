"""Build the committed demo sample ``data/tng300_128.npz`` from the original
``.fvol`` volumes (IllustrisTNG TNG300-3-Dark, snapshot 99, z = 0, on a 128^3 grid).

    python scripts/prepare_sample.py --raw path/to/fvols

Fields are stored as density relative to their own mean (rho / rho_bar) in
float16, which keeps the file small and loses nothing that matters after the
log transform the model applies. The mask is stored as uint8 class IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvdemo.data import check_split, make_split  # noqa: E402
from dvdemo.io import CLASS_NAMES, read_fvol, read_mask  # noqa: E402

FILES = {
    "dm": ("DM_DEN_snap99_Nm=128.fvol", 0.33,
           "dark-matter particle density (all particles)"),
    "subhalo_L7": ("subs1_mass_Nm128_L7_d_None_smooth.fvol", 7.0,
                   "subhalo mass density, subsampled to mean separation 7 h⁻¹ Mpc"),
}
MASK = "TNG300-3-Dark-mask-Nm=128-th=0.65-sig=0.6.fvol"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="folder with the .fvol files")
    ap.add_argument("--out", default="data/tng300_128.npz")
    args = ap.parse_args()

    mask = read_mask(os.path.join(args.raw, MASK))
    arrays = {"mask": mask}
    fields, sources = {}, {MASK: sha256(os.path.join(args.raw, MASK))}
    for key, (fname, lam, desc) in FILES.items():
        path = os.path.join(args.raw, fname)
        if not os.path.exists(path):
            print(f"skipping {key}: {fname} not found")
            continue
        vol = read_fvol(path)
        if vol.shape != mask.shape:
            raise ValueError(f"{fname} shape {vol.shape} != mask shape {mask.shape}")
        if vol.min() < 0:
            raise ValueError(f"{fname} has negative values; expected a density")
        rel = vol / vol.mean()
        if rel.max() > 6e4:
            raise ValueError(f"{fname}: rho/rho_bar max {rel.max():.3g} overflows float16")
        arrays[key] = rel.astype(np.float16)
        fields[key] = {"lambda_mpc_h": lam, "description": desc, "units": "rho / mean(rho)",
                       "empty_voxel_fraction": float((vol == 0).mean())}
        sources[fname] = sha256(path)
        print(f"{key:12s} rho/rho_bar in [{rel.min():.3g}, {rel.max():.3g}], "
              f"empty voxels {100 * (vol == 0).mean():.1f}%")

    split = make_split(mask.shape[0])
    check_split(split, mask.shape[0])
    meta = {
        "simulation": "IllustrisTNG TNG300-3-Dark, snapshot 99 (z = 0)",
        "box_mpc_h": 205.0,
        "grid": list(mask.shape),
        "voxel_mpc_h": 205.0 / mask.shape[0],
        "labels": {"method": "T-web (tidal tensor eigenvalues of the smoothed DM field)",
                   "eigenvalue_threshold": 0.65, "smoothing_sigma_grid": 0.6,
                   "classes": list(CLASS_NAMES),
                   "fractions": (np.bincount(mask.ravel(), minlength=4) / mask.size).tolist()},
        "fields": fields,
        "split": dict(split, note="32^3 subcubes split at random into training and validation "
                                   "(80/20, seed 12); only training subcubes are augmented"),
        "source_sha256": sources,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
