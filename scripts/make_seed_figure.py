"""Make ``assets/seed_study.png`` from ``results/seed_study.json`` (written by
notebooks/seed_study.ipynb). Needs only NumPy/Matplotlib."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvdemo.viz import plot_seed_study  # noqa: E402

LABELS = {"dm_model": "dm_model: dark matter → dark matter",
          "subhalo_from_scratch": "subhalo_from_scratch: trained on galaxies only",
          "subhalo_curriculum": "subhalo_curriculum: dm_model fine-tuned on galaxies"}


def main():
    with open("results/seed_study.json") as f:
        study = json.load(f)
    runs = study["runs"]
    fig = plot_seed_study(runs, LABELS)
    fig.savefig("assets/seed_study.png", dpi=150, bbox_inches="tight")
    d = np.array([r["subhalo_curriculum"]["mcc"] - r["subhalo_from_scratch"]["mcc"] for r in runs])
    print(f"{len(runs)} seeds; curriculum - scratch MCC: mean {d.mean():+.4f}, "
          f"std {d.std(ddof=1):.4f}, curriculum ahead in {(d > 0).sum()}/{len(d)}")
    for m in list(LABELS) + ["subhalo_no_retraining"]:
        for k in ("void_f1", "mcc"):
            v = np.array([r[m][k] for r in runs])
            print(f"  {m:24s} {k:8s} {v.mean():.4f} ± {v.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
