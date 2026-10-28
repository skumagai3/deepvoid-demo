"""Draw ``assets/architecture.png``: the demo's 3D U-Net as built by
``dvdemo.model.build_unet(depth=3, filters=16)``, for a 32^3 training subcube."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DEPTH, FILTERS, SIDE = 3, 16, 32
ENC, DEC, BOT, IO = "#86a9d6", "#e7a43a", "#2f5d8a", "#d9d9d9"


def box(ax, x, y, w, h, color, text, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444", lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=tc, linespacing=1.3)


def arrow(ax, a, b, color="#444", style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=color, lw=1.1,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-0.9, 5.0); ax.axis("off")
    w, h = 1.5, 0.75
    xs_enc = [1.4, 2.4, 3.4]
    xs_dec = [8.6, 7.6, 6.6]
    ys = [4.0, 2.9, 1.8]
    box(ax, 0.0, 4.0, 1.1, h, IO, f"input\n{SIDE}³ × 1\ndensity")
    box(ax, 10.0, 4.0, 1.1, h, IO, f"output\n{SIDE}³ × 4\nP(class)")
    arrow(ax, (0.55, 4.0), (xs_enc[0] - w / 2, ys[0]))
    for d in range(DEPTH):
        side, f = SIDE // 2 ** d, FILTERS * 2 ** d
        box(ax, xs_enc[d], ys[d], w, h, ENC, f"encoder {d}\n{side}³ × {f}")
        box(ax, xs_dec[d], ys[d], w, h, DEC, f"decoder {d}\n{side}³ × {f}")
        arrow(ax, (xs_enc[d] + w / 2, ys[d]), (xs_dec[d] - w / 2, ys[d]), color="#999", ls="--")
        ax.text((xs_enc[d] + xs_dec[d]) / 2, ys[d] + 0.08,
                "skip connection (concatenate)" if d == 0 else "skip",
                ha="center", va="bottom", fontsize=7, color="#777")
    for d in range(DEPTH - 1):
        arrow(ax, (xs_enc[d], ys[d] - h / 2), (xs_enc[d + 1], ys[d + 1] + h / 2))
        arrow(ax, (xs_dec[d + 1], ys[d + 1] + h / 2), (xs_dec[d], ys[d] - h / 2))
    bside, bf = SIDE // 2 ** DEPTH, FILTERS * 2 ** DEPTH
    box(ax, 5.0, 0.7, 1.7, h, BOT, f"bottleneck\n{bside}³ × {bf}", tc="white")
    arrow(ax, (xs_enc[-1], ys[-1] - h / 2), (5.0 - 0.85, 0.7))
    arrow(ax, (5.0 + 0.85, 0.7), (xs_dec[-1], ys[-1] - h / 2))
    arrow(ax, (xs_dec[0] + w / 2, ys[0]), (10.0 - 0.55, 4.0))
    ax.text(1.55, 1.25, "max-pool ↓2", fontsize=8, color="#444", ha="center")
    ax.text(8.45, 1.25, "transposed conv ↑2", fontsize=8, color="#444", ha="center")
    ax.text(5.0, -0.45,
            "each block: 2 × (3³ convolution → batch norm → ReLU)   ·   output: 1³ convolution → softmax over "
            "void / wall / filament / halo\n1.40 M parameters · fully convolutional, so at prediction time it runs "
            "on overlapping windows of the full 128³ box",
            ha="center", va="center", fontsize=8.5, color="#333", linespacing=1.5)
    ax.text(5.0, 4.75, "The demo's 3D U-Net (shapes for one 32³ training subcube: side³ × channels)",
            ha="center", fontsize=11)
    os.makedirs("assets", exist_ok=True)
    fig.savefig("assets/architecture.png", dpi=150, bbox_inches="tight")
    print("wrote assets/architecture.png")


if __name__ == "__main__":
    main()
