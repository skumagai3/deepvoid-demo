"""Make ``assets/alignment.png``: dark matter, subhalos and the mask on slices through x, y and z."""
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvdemo.io import load_sample  # noqa: E402
from dvdemo.viz import alignment_figure  # noqa: E402

s = load_sample("data/tng300_128.npz")

fig = alignment_figure(s, 64)
fig.savefig("assets/alignment.png", dpi=120, bbox_inches="tight")
print("wrote assets/alignment.png")
