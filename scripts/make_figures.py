"""Make the README figures and the hero GIF from saved predictions.

    python scripts/make_figures.py            # after training (needs results/ and weights/*.json)

Needs only NumPy/Matplotlib/Pillow: predictions are read from
``results/predictions.npz`` and metrics from ``weights/*.json``.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvdemo.io import load_sample  # noqa: E402
from dvdemo.viz import (compare_slice, plot_confusion, plot_model_comparison,  # noqa: E402
                        render_sweep, save_gif)

OUT = "assets"
LABELS = {"dm_model": "dm_model: dark matter → dark matter",
          "subhalo_no_retraining": "subhalo_no_retraining: dm_model applied to galaxies",
          "subhalo_from_scratch": "subhalo_from_scratch: trained on galaxies only",
          "subhalo_curriculum": "subhalo_curriculum: dm_model fine-tuned on galaxies"}


def main():
    os.makedirs(OUT, exist_ok=True)
    s = load_sample("data/tng300_128.npz")
    preds = dict(np.load("results/predictions.npz"))
    runs = {}
    for name in ("dm_model", "subhalo_curriculum", "subhalo_from_scratch"):
        with open(f"weights/{name}.json") as f:
            runs[name] = json.load(f)
    with open("results/metrics.json") as f:
        val = json.load(f)["validation"]
    split = s["meta"]["split"]
    mid = 48   # x = 32-63 contains 6 of the 13 validation subcubes
    header = ("slice x = {pos:.0f} h⁻¹ Mpc   ·   dashed: validation subcubes "
              "(never trained on; all scores come from these)")

    # Hero GIF: sweep through the whole box, dense and sparse rows.
    frames = render_sweep(
        [("Dark matter, λ = 0.33 h⁻¹ Mpc", s["dm"], False),
         ("Galaxies, λ = 7 h⁻¹ Mpc", s["subhalo_L7"], True)],
        s["mask"], [preds["dm_model"], preds["subhalo_curriculum"]],
        range(0, 128, 4), header=header, split=split)
    save_gif(frames, f"{OUT}/hero.gif", fps=2)   # 0.5 s per slice

    fig = compare_slice(s["dm"], s["mask"], preds["dm_model"], mid,
                        field_title="Dark matter (λ = 0.33 h⁻¹ Mpc)", split=split)
    fig.savefig(f"{OUT}/dense_slice.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    fig = compare_slice(s["subhalo_L7"], s["mask"], preds["subhalo_curriculum"], mid,
                        field_title="Galaxies (λ = 7 h⁻¹ Mpc)", sparse=True, split=split)
    fig.savefig(f"{OUT}/sparse_slice.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    plot_model_comparison(val, LABELS, ax=ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/model_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    plot_confusion(runs["dm_model"]["val_metrics"]["confusion_matrix"], ax=axes[0],
                   title="dm_model (λ = 0.33 h⁻¹ Mpc)")
    plot_confusion(runs["subhalo_curriculum"]["val_metrics"]["confusion_matrix"], ax=axes[1],
                   title="subhalo_curriculum (λ = 7 h⁻¹ Mpc)")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion.png", dpi=150); plt.close(fig)

    # Validation loss per epoch; the curriculum continues from the dense model.
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    n_dense = len(runs["dm_model"]["train"]["history"]["val_loss"])
    for name, color, start in (("dm_model", "#2f5d8a", 0), ("subhalo_curriculum", "#e7a43a", n_dense),
                               ("subhalo_from_scratch", "#999999", 0)):
        v = runs[name]["train"]["history"]["val_loss"]
        ax.plot(np.arange(1, len(v) + 1) + start, v, color=color, label=LABELS[name])
    ax.axvline(n_dense, color="#bbb", lw=1, ls="--")
    ax.text(n_dense, ax.get_ylim()[0], "  switch to galaxies", fontsize=8, color="#777", va="bottom")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curves.png", dpi=150); plt.close(fig)

    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if os.path.isfile(p):
            print(f"{p:32s} {os.path.getsize(p) / 1e3:8.0f} kB")


if __name__ == "__main__":
    main()
