"""Plotting: slice comparisons, confusion matrices, model comparison, GIFs."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import CLASS_NAMES

# Voids pale, the web in color: the eye reads filaments and halos as structure.
CLASS_COLORS = ("#e9eef5", "#86a9d6", "#e7a43a", "#b8322a")
ERROR_COLORS = ("#d9d9d9", "#c0392b")     # correct, wrong

# Physical scale of the demo box (IllustrisTNG300: 205 h^-1 Mpc, periodic).
BOX = 205.0
UNIT = "h⁻¹ Mpc"
PLANES = {0: ("y", "z"), 1: ("x", "z"), 2: ("x", "y")}   # (horizontal, vertical) per slice axis


def position(index: int, n: int = 128, box: float = BOX) -> float:
    """Center of voxel ``index`` in h^-1 Mpc."""
    return (index + 0.5) * box / n


def _extent(box):
    return (0, box, 0, box)


def class_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(CLASS_COLORS, name="tweb")
    return cmap, BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)


def take_slice(vol: np.ndarray, index: int, axis: int = 0) -> np.ndarray:
    return np.take(vol, index, axis=axis)


def show_density(ax, field2d: np.ndarray, title: str = "", sparse: bool = False, box: float = BOX):
    """log10(1 + rho/rho_bar), so empty voxels of a sparse field stay finite.
    Slices are 2D arrays indexed [horizontal, vertical]; axes are in h^-1 Mpc."""
    img = np.log10(1.0 + np.asarray(field2d, dtype=np.float64))
    vmax = np.percentile(img, 99.5) if sparse else np.percentile(img, 99.8)
    ax.imshow(img.T, origin="lower", cmap="magma", vmin=0, vmax=max(vmax, 1e-6),
              interpolation="nearest", extent=_extent(box))
    _style(ax, title)


def show_labels(ax, labels2d: np.ndarray, title: str = "", box: float = BOX):
    cmap, norm = class_cmap()
    ax.imshow(np.asarray(labels2d).T, origin="lower", cmap=cmap, norm=norm,
              interpolation="nearest", extent=_extent(box))
    _style(ax, title)


def show_errors(ax, truth2d: np.ndarray, pred2d: np.ndarray, title: str = "", box: float = BOX):
    from matplotlib.colors import ListedColormap
    wrong = (np.asarray(truth2d) != np.asarray(pred2d)).astype(int)
    ax.imshow(wrong.T, origin="lower", cmap=ListedColormap(ERROR_COLORS), vmin=0, vmax=1,
              interpolation="nearest", extent=_extent(box))
    _style(ax, f"{title} ({100 * wrong.mean():.0f}% wrong)" if title else "")


def _style(ax, title):
    ax.set_xticks(np.arange(0, 201, 50)); ax.set_yticks(np.arange(0, 201, 50))
    ax.tick_params(labelsize=8, length=2.5, color="#888")
    for s in ax.spines.values():
        s.set_color("#bbb"); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=11)


def label_axes(ax, axis: int = 0, x: bool = True, y: bool = True, prefix: str = ""):
    """Physical axis labels for a slice through ``axis``."""
    h, v = PLANES[axis]
    if x:
        ax.set_xlabel(f"{h} [{UNIT}]", fontsize=9, labelpad=2)
    if y:
        ax.set_ylabel(f"{prefix}{v} [{UNIT}]", fontsize=9, labelpad=2)


def class_legend(fig, loc="lower center", ncol=4, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#888", label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    return fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False, fontsize=10, **kw)


def compare_slice(field: np.ndarray, truth: np.ndarray, pred: np.ndarray, index: int,
                  axis: int = 0, field_title: str = "Input density", sparse: bool = False,
                  errors: bool = True, figsize=None, split: Optional[Dict] = None):
    """Input | T-web truth | prediction (| errors) for one slice. With ``split``,
    validation subcubes on the slice are outlined (dashed)."""
    import matplotlib.pyplot as plt
    n = 4 if errors else 3
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.5 * n, 4.4))
    f2, t2, p2 = (take_slice(v, index, axis) for v in (field, truth, pred))
    show_density(axes[0], f2, field_title, sparse=sparse)
    show_labels(axes[1], t2, "Ground truth (T-web)")
    show_labels(axes[2], p2, "U-Net prediction")
    if errors:
        show_errors(axes[3], t2, p2, "Errors")
    for i, a in enumerate(axes):
        label_axes(a, axis, y=(i == 0))
        if split is not None and i > 0:
            outline_val_cells(a, split, index, axis, n=field.shape[axis])
    note = "   ·   dashed: validation subcubes (never trained on)" if split is not None else ""
    fig.suptitle(f"slice {'xyz'[axis]} = {position(index, field.shape[axis]):.0f} {UNIT}{note}",
                 fontsize=10, color="#555", y=0.99)
    class_legend(fig, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def plot_confusion(cm, names: Sequence[str] = CLASS_NAMES, ax=None, title: str = ""):
    """Row-normalized confusion matrix (each row: where that true class went)."""
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.55 else "#222")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def plot_model_comparison(metrics: Dict[str, Dict[str, float]], labels: Dict[str, str],
                          keys=("void_f1", "mcc"), key_names=("Void F1", "MCC"), ax=None):
    """Grouped horizontal bars of held-out metrics for several models."""
    import matplotlib.pyplot as plt
    models = list(labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.55 * len(models) * len(keys) + 1.2))
    h = 0.8 / len(keys)
    colors = ("#2f5d8a", "#9bb8d3")
    for j, (k, kn) in enumerate(zip(keys, key_names)):
        vals = [metrics[m][k] for m in models]
        ys = np.arange(len(models)) + (j - (len(keys) - 1) / 2) * h
        bars = ax.barh(ys, vals, height=h * 0.92, color=colors[j % 2], label=kn)
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([labels[m] for m in models])
    ax.invert_yaxis(); ax.set_xlim(0, 1.1); ax.set_xlabel("score on the validation subcubes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(keys))
    return ax


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def render_sweep(rows: List[Tuple[str, np.ndarray, bool]], truth: np.ndarray,
                 preds: List[np.ndarray], indices: Sequence[int], axis: int = 0,
                 dpi: int = 90, header: Optional[str] = None,
                 split: Optional[Dict] = None) -> List[np.ndarray]:
    """Frames for a slice-sweep GIF. Each row is (row label, input field,
    sparse?) paired with ``preds[row]``; columns are input | truth | prediction."""
    import matplotlib.pyplot as plt
    frames = []
    for idx in indices:
        fig, axes = plt.subplots(len(rows), 3, figsize=(9.8, 3.2 * len(rows) + 1.1), dpi=dpi,
                                 squeeze=False)
        for r, ((label, field, sparse), pred) in enumerate(zip(rows, preds)):
            top = r == 0
            show_density(axes[r, 0], take_slice(field, idx, axis), "Input" if top else "", sparse=sparse)
            show_labels(axes[r, 1], take_slice(truth, idx, axis), "Ground truth (T-web)" if top else "")
            show_labels(axes[r, 2], take_slice(pred, idx, axis), "U-Net prediction" if top else "")
            last = r == len(rows) - 1
            if split is not None:
                for c in (1, 2):
                    outline_val_cells(axes[r, c], split, idx, axis, n=truth.shape[axis])
            for c in range(3):
                label_axes(axes[r, c], axis, x=last, y=(c == 0),
                           prefix=f"{label}\n" if c == 0 else "")
        if header:
            pos = position(idx, truth.shape[axis])
            fig.suptitle(header.format(index=idx, pos=pos), fontsize=10, y=0.99, color="#555")
        class_legend(fig, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=1.2)
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    return frames


def save_gif(frames: List[np.ndarray], path: str, fps: float = 6, colors: int = 64) -> None:
    """Write an optimized, looping GIF with a shared palette."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    palette = imgs[len(imgs) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=True, disposal=2)


VAL_COLOR = "#b8322a"


def outline_val_cells(ax, split: Dict, index: int, axis: int = 0, n: int = 128,
                      color: str = VAL_COLOR, lw: float = 1.3, fill: bool = False):
    """Dashed squares around (or, with ``fill``, shading over) the validation
    subcubes a slice passes through."""
    from matplotlib.patches import Rectangle
    from .data import val_cells_in_slice
    s = split["subgrid"]
    cell = BOX / n * s
    h, v = [d for d in range(3) if d != axis]
    for c in val_cells_in_slice(split, index, axis):
        if fill:
            ax.add_patch(Rectangle((c[h] * cell, c[v] * cell), cell, cell, facecolor=color,
                                   alpha=0.35, edgecolor="none"))
        ax.add_patch(Rectangle((c[h] * cell, c[v] * cell), cell, cell, fill=False,
                               edgecolor=color, lw=lw, ls="-" if fill else (0, (4, 2))))


def show_split(ax, mask: np.ndarray, split: Dict, index: int = 48, axis: int = 0):
    """One slice of the labels with the 32^3 subcube grid drawn on it and the
    validation subcubes it passes through marked."""
    show_labels(ax, take_slice(mask, index, axis))
    cell = BOX / mask.shape[0] * split["subgrid"]
    for t in np.arange(cell, BOX - 1e-6, cell):
        ax.axvline(t, color="#777", lw=0.5); ax.axhline(t, color="#777", lw=0.5)
    outline_val_cells(ax, split, index, axis, n=mask.shape[0], lw=1.5, fill=True)
    n_here = len([c for c in split["val"] if c[axis] == index // split["subgrid"]])
    ax.set_title(f"slice {'xyz'[axis]} = {position(index, mask.shape[axis]):.0f} {UNIT}: "
                 f"{n_here} of the {len(split['val'])} validation subcubes (shaded)", fontsize=10)
    label_axes(ax, axis)
    return ax


def _contour(ax, labels2d, level, color, lw=0.7, box: float = BOX):
    arr = np.asarray(labels2d)
    c = (np.arange(arr.shape[0]) + 0.5) * box / arr.shape[0]
    ax.contour(c, c, (arr.T >= level).astype(float), levels=[0.5], colors=color, linewidths=lw)


def alignment_figure(sample: dict, index: int = 64):
    """Dark matter, subhalos and the T-web mask on the same slice, for slices
    through each of the three axes (rows). The mask's void edges are drawn on
    both inputs, so you can see the inputs and labels line up."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 3, figsize=(12.5, 12.8))
    for axis in range(3):
        m2 = take_slice(sample["mask"], index, axis)
        row = ax[axis]
        show_density(row[0], take_slice(sample["dm"], index, axis),
                     f"Dark matter (λ = 0.33 {UNIT})" if axis == 0 else "")
        show_density(row[1], take_slice(sample["subhalo_L7"], index, axis),
                     f"Subhalos (λ = 7 {UNIT})" if axis == 0 else "", sparse=True)
        for a in row[:2]:
            _contour(a, m2, 1, "#9fd3ff", lw=0.6)
        show_labels(row[2], m2, "T-web mask (ground truth)" if axis == 0 else "")
        pos = position(index, sample["mask"].shape[axis])
        h, v = PLANES[axis]
        for c, a in enumerate(row):
            # tick numbers only on the outer panels: every panel spans 0-205 h^-1 Mpc
            a.tick_params(labelbottom=(axis == 2), labelleft=(c == 0))
        row[0].set_ylabel(f"slice {'xyz'[axis]} = {pos:.0f} {UNIT}  ({h}–{v} plane)\n\n"
                          f"{v} [{UNIT}]", fontsize=9, labelpad=2)
    for a in ax[2]:
        a.set_xlabel(f"x [{UNIT}]", fontsize=9, labelpad=2)
    class_legend(fig, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.035, "blue lines: edges of the mask's voids, drawn on both inputs",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
    return fig


def plot_seed_study(runs: List[Dict[str, Dict[str, float]]], labels: Dict[str, str],
                    keys=("void_f1", "mcc"), key_names=("Void F1", "MCC"),
                    pair=("subhalo_curriculum", "subhalo_from_scratch")):
    """Scores across training seeds: each seed as a dot, mean ± standard deviation
    as a bar, per model; plus a panel with the per-seed difference between the two
    models in ``pair`` (curriculum - from scratch) on MCC."""
    import matplotlib.pyplot as plt
    models = list(labels)
    n = len(runs)
    fig, axes = plt.subplots(1, len(keys) + 1, figsize=(4.4 * (len(keys) + 1) + 2.5, 0.75 * len(models) + 1.7),
                             gridspec_kw={"width_ratios": [1] * len(keys) + [0.8]})
    rng = np.random.default_rng(0)
    sd = (lambda v: v.std(ddof=1)) if n > 1 else (lambda v: 0.0)
    for ax, k, kn in zip(axes[:-1], keys, key_names):
        for i, m in enumerate(models):
            v = np.array([r[m][k] for r in runs])
            ax.scatter(v, i + rng.uniform(-0.12, 0.12, n), s=16, color="#9bb8d3", zorder=2)
            ax.errorbar(v.mean(), i, xerr=sd(v), fmt="o", color="#2f5d8a", ms=6, capsize=4,
                        lw=1.6, zorder=3)
            ax.annotate(f"{v.mean():.3f} ± {sd(v):.3f}", (v.mean(), i), xytext=(0, -14),
                        textcoords="offset points", ha="center", va="top", fontsize=8, color="#333")
        ax.set_title(kn, fontsize=11)
        ax.set_xlabel(f"validation {kn}")
        ax.set_ylim(len(models) - 0.4, -0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#eee", zorder=0)
        ax.margins(x=0.25)
    axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels([labels[m] for m in models])
    for ax in axes[1:-1]:
        ax.set_yticks([])
    d = np.array([r[pair[0]]["mcc"] - r[pair[1]]["mcc"] for r in runs])
    ax = axes[-1]
    ax.axvline(0, color="#999", lw=1)
    ax.scatter(d, rng.uniform(-0.12, 0.12, n), s=20, color="#e7a43a", zorder=2)
    ax.errorbar(d.mean(), 0, xerr=sd(d), fmt="o", color="#b8322a", ms=6, capsize=4, lw=1.6, zorder=3)
    ax.annotate(f"{d.mean():+.3f} ± {sd(d):.3f}\nahead in {(d > 0).sum()} of {n}", (d.mean(), 0),
                xytext=(0, -16), textcoords="offset points", ha="center", va="top", fontsize=8)
    lim = max(0.03, np.abs(d).max() * 1.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(1, -1); ax.set_yticks([])
    ax.set_title("curriculum − from scratch", fontsize=11)
    ax.set_xlabel("difference in validation MCC")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(f"{n} training seeds, same split and settings", fontsize=10, color="#555", y=1.0)
    fig.tight_layout()
    return fig
