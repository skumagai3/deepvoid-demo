"""Generate the two notebooks in ``notebooks/`` (run from the repo root).

Keeping the notebooks as Python source makes them easy to review and diff;
the generated ``.ipynb`` files are committed without outputs.
"""
import json
import os

REPO = "skumagai3/deepvoid-demo"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, gpu=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def badge(path):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{path})")


SETUP = f'''
# Get the code, data and weights. In Colab this clones the repo (about 20 MB);
# locally, run the notebook from inside the repo.
import os, subprocess
if not os.path.isdir("dvdemo"):
    if os.path.isdir("../dvdemo"):
        os.chdir("..")
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/{REPO}.git"], check=True)
        os.chdir("deepvoid-demo")
print("working in", os.getcwd())
'''

MODELS_TABLE = """
| Model | Trained on | Applied to | What it tests |
|---|---|---|---|
| `dm_model` | dark matter (λ = 0.33 h⁻¹ Mpc) | dark matter | the basic task, with the dark matter density field as input |
| `subhalo_no_retraining` | dark matter | galaxies (λ = 7 h⁻¹ Mpc) | does a dark-matter model work on sparse tracers as-is? (input rescaled, weights unchanged) |
| `subhalo_from_scratch` | galaxies | galaxies | training directly on the sparse field, starting from random weights |
| `subhalo_curriculum` | dark matter, then galaxies | galaxies | one-step curriculum learning: start from `dm_model`, freeze its first encoder block, fine-tune on galaxies |

All four are scored against the same T-web labels (computed from the full dark matter) on the same
validation subcubes.
"""

DEMO = [
    md(f"""
# DeepVoid: segmenting the cosmic web with a 3D U-Net

{badge("notebooks/deepvoid_demo.ipynb")}

This notebook runs trained DeepVoid models end to end, in a few minutes on a CPU (under a minute on a GPU):

1. load a 128³ cube from the IllustrisTNG300 dark-matter simulation (205 h⁻¹ Mpc on a side),
2. classify every one of its 2.1 million voxels as void, wall, filament or halo,
3. compare the prediction with the ground truth on validation subcubes the model never trained on,
4. repeat with only the galaxies as input, which is closer to what a galaxy survey sees.

The labels come from the T-web: at each point, count how many directions the local tidal field is
compressing matter (0 = void, 1 = wall, 2 = filament, 3 = halo). The network only sees the density
field and has to learn this from it.

*Simulation data: [IllustrisTNG](https://www.tng-project.org/) (TNG300-3-Dark), produced and publicly
released by the IllustrisTNG collaboration. See the README for the papers to cite.*
"""),
    code(SETUP),
    code('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from dvdemo import viz
from dvdemo.data import Normalizer, split_masks
from dvdemo.inference import keras_predict_fn, predict_volume
from dvdemo.io import CLASS_NAMES, load_sample
from dvdemo.metrics import evaluate, format_report
from dvdemo.train import load_run

print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "none (CPU is fine)")
'''),
    md("""
## 1. The data

The sample is 4.5 MB: two input fields and the label cube, all on the same 128³ grid (1.6 h⁻¹ Mpc voxels).

* Dark matter: the dark matter density, sampled by particles 0.33 h⁻¹ Mpc apart. Lots of detail, but not something we can observe.
* Galaxies at λ = 7 h⁻¹ Mpc. This simulation only contains dark matter, so the galaxies are its subhalos:
  the small, dense clumps of dark matter where galaxies would form. They're close enough that I call them
  galaxies (the code and file names say `subhalo`). They're thinned to a mean separation of 7 h⁻¹ Mpc, about
  10,000 times fewer tracers than the dark matter, and 93% of voxels are empty, which is much closer to what a galaxy survey sees.
* Labels: T-web classes computed from the full dark matter, used for both inputs. With the sparse input, the
  network has to infer structure it can't see directly.
"""),
    code('''
sample = load_sample("data/tng300_128.npz")
meta, split, mask = sample["meta"], sample["meta"]["split"], sample["mask"]
print(meta["simulation"])
print(f"box {meta['box_mpc_h']:.0f} h⁻¹ Mpc, grid {meta['grid'][0]}^3, voxel {meta['voxel_mpc_h']:.2f} h⁻¹ Mpc")
for k, f in meta["fields"].items():
    print(f"  {k:11s} {f['description']}  ({100 * f['empty_voxel_fraction']:.0f}% empty voxels)")
print("class fractions:", {c: round(v, 3) for c, v in zip(CLASS_NAMES, meta["labels"]["fractions"])})

TRAIN, VAL = split_masks(mask.shape, split)   # VAL: the voxels used for every score below
x_mid = 48                                    # this x-slice passes through 6 of the 13 validation subcubes

fig = viz.alignment_figure(sample, 64)   # middle slice through each axis
plt.show()
'''),
    md("""
Voids fill most of the volume (62%) while halos are only 0.5% of voxels, so plain accuracy is misleading:
predicting "void" everywhere already scores 62%. I report void F1, since voids are the class I care about
most, and the multiclass Matthews correlation coefficient (MCC), which accounts for the class imbalance.

### How the box is split

The box is cut into 32³ subcubes (4 × 4 × 4 = 64) and the subcubes are
split at random into training (51) and validation (13). Only the training subcubes are augmented, with four
copies each (as-is and rotated 90° about each axis); validation subcubes are never rotated or trained on.
Validation loss drives learning-rate reduction and early stopping, and every number in this notebook is
computed on the validation subcubes (13 × 32³ ≈ 426,000 voxels, 20% of the box).
"""),
    code('''
fig, ax = plt.subplots(figsize=(5.4, 5.6))
viz.show_split(ax, mask, split, index=x_mid); plt.show()
'''),
    md("""
## 2. Predict the full box with the dark-matter model

The model is a 3D U-Net (3 resolution levels, 16 → 128 filters, about 1.4 M parameters) trained on the
augmented 32³ training subcubes. The input scaling (min-max by default; the training notebook explains the
alternatives) is fitted on the training voxels and saved with the model.

At prediction time the periodic box is covered by 512 overlapping 32³ windows; only the central 16³ of
each is kept, so every voxel gets 8 voxels of context on all sides and there are no seams.
"""),
    code('''
model, norm, record = load_run("weights/dm_model")
print(f"{record['params']:,} parameters; input scaling: {norm.method}; "
      f"best epoch {record['train']['best_epoch']} of {record['train']['epochs_run']}")

t = time.time()
probs_dense = predict_volume(keras_predict_fn(model), norm.transform(sample["dm"]), verbose=True)
pred_dense = probs_dense.argmax(-1).astype(np.uint8)
print(f"full 128^3 box predicted in {time.time() - t:.1f} s")
'''),
    code('''
m_dense = evaluate(mask, pred_dense, list(CLASS_NAMES), where=VAL)
print(format_report(m_dense, CLASS_NAMES))
viz.plot_confusion(m_dense["confusion_matrix"], title="Validation subcubes, dark-matter input"); plt.show()
'''),
    code('''
fig = viz.compare_slice(sample["dm"], mask, pred_dense, x_mid, field_title="Dark matter, λ = 0.33 h⁻¹ Mpc",
                        split=split)
plt.show()
'''),
    md("""
The dashed squares are the validation subcubes on this slice. Compare where the errors fall with the
class boundaries in the ground truth. The maps below show the model's probability of "void" and its
overall uncertainty (entropy of the predicted class probabilities) on the same slice.
"""),
    code('''
p_void = probs_dense[x_mid, :, :, 0]
entropy = -(probs_dense[x_mid] * np.log(np.clip(probs_dense[x_mid], 1e-7, 1))).sum(-1)
ext = (0, viz.BOX, 0, viz.BOX)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
im = axes[0].imshow(p_void.T, origin="lower", cmap="Blues_r", vmin=0, vmax=1, extent=ext)
axes[0].set_title("P(void)"); plt.colorbar(im, ax=axes[0], fraction=0.046)
im = axes[1].imshow(entropy.T, origin="lower", cmap="viridis", extent=ext)
axes[1].set_title("Prediction entropy (uncertainty)"); plt.colorbar(im, ax=axes[1], fraction=0.046)
for a in axes: viz.label_axes(a, axis=0)
plt.tight_layout(); plt.show()
'''),
    md("Browse the box slice by slice. Dashed squares mark the validation subcubes on each slice; everything outside them was used for training."),
    code('''
from ipywidgets import IntSlider, interact

@interact(x=IntSlider(value=x_mid, min=0, max=mask.shape[0] - 1, description="slice x"))
def browse(x):
    viz.compare_slice(sample["dm"], mask, pred_dense, x,
                      field_title="Dark matter, λ = 0.33 h⁻¹ Mpc", split=split)
    plt.show()
'''),
    md(f"""
## 3. Sparse galaxies (λ = 7 h⁻¹ Mpc)

Real surveys see galaxies, not dark matter. Here the input is only the galaxy field, but the target is
still the T-web computed from the dark matter. I compare three ways to handle it, alongside the
dark matter model from section 2:

{MODELS_TABLE}
All models use the same split, training settings and early stopping.
"""),
    code('''
preds, scores = {}, {}
for name in ("subhalo_curriculum", "subhalo_from_scratch"):
    mdl, nrm, rec = load_run(f"weights/{name}")
    p = predict_volume(keras_predict_fn(mdl), nrm.transform(sample["subhalo_L7"]))
    preds[name] = p.argmax(-1).astype(np.uint8)

zs = Normalizer(norm.method).fit(sample["subhalo_L7"][TRAIN])
preds["subhalo_no_retraining"] = predict_volume(keras_predict_fn(model), zs.transform(sample["subhalo_L7"])).argmax(-1).astype(np.uint8)

labels = {"dm_model": "dm_model: dark matter → dark matter",
          "subhalo_no_retraining": "subhalo_no_retraining: dm_model applied to galaxies",
          "subhalo_from_scratch": "subhalo_from_scratch: trained on galaxies only",
          "subhalo_curriculum": "subhalo_curriculum: dm_model fine-tuned on galaxies"}
scores["dm_model"] = m_dense
for k in ("subhalo_no_retraining", "subhalo_from_scratch", "subhalo_curriculum"):
    scores[k] = evaluate(mask, preds[k], list(CLASS_NAMES), where=VAL)

cols = ["void_f1", "mcc", "balanced_accuracy", "wall_f1", "filament_f1", "halo_f1"]
table = pd.DataFrame({labels[k]: {c: scores[k][c] for c in cols} for k in labels}).T
table.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=1)
'''),
    code('''
fig, ax = plt.subplots(figsize=(7.4, 3.8))
viz.plot_model_comparison(scores, labels, ax=ax); plt.tight_layout(); plt.show()
fig = viz.compare_slice(sample["subhalo_L7"], mask, preds["subhalo_curriculum"], x_mid,
                        field_title="Galaxies, λ = 7 h⁻¹ Mpc", sparse=True, split=split)
plt.show()
'''),
    md("""
## Caveats

* This is a scaled-down version of the method. The full-scale results (Kumagai et al., ApJ 2026) are
  void F1 = 0.96 for dark matter at λ = 0.33 h⁻¹ Mpc and void F1 = 0.89 for galaxies at λ = 10 h⁻¹ Mpc.
  To fit in a repo and train in minutes, this demo uses a 4× coarser grid (1.6 instead of 0.4 h⁻¹ Mpc
  voxels), 64× less data, a small U-Net and very little tuning, so its numbers aren't directly
  comparable. The README has more detail.
* The curriculum is simplified. The demo goes from dark matter (λ = 0.33) straight to galaxies at
  λ = 7 h⁻¹ Mpc in one step. The paper's best scheme is two-step (λ = 0.33 → 3 → 10 h⁻¹ Mpc) on much
  sparser tracers in voxel terms, which is where pretraining should help most. Even there the gain over
  direct training is modest (MCC 0.60 vs. 0.56, same void F1 of 0.89), so a near-tie here isn't surprising.
* It's one simulation. Good scores here show the model generalizes to unseen regions of the same
  simulation, not to other cosmologies or to real surveys (which add redshift-space distortions, survey masks
  and selection effects).
* This is a segmentation, not a void catalog. Turning voxel labels into individual voids (centers, radii)
  needs a separate, well-defined post-processing step.

Links: [paper (arXiv:2504.21134)](https://arxiv.org/abs/2504.21134) ·
[research code](https://github.com/skumagai3/deepvoid_misc) ·
[DV_NEW rewrite](https://github.com/skumagai3/DV_NEW) ·
[how the demo models were trained](train_demo_models.ipynb)
"""),
]


TRAIN = [
    md(f"""
# Train the DeepVoid demo models

{badge("notebooks/train_demo_models.ipynb")}

Reproduces everything in `weights/` and `results/`, and the figures in `assets/`. Each training run
produces these models:

{MODELS_TABLE}

Training setup: 32³ subcubes, an 80/20 random split of subcubes (seed 12), the four
rotations on training subcubes only, Adam, sparse categorical cross-entropy, learning-rate reduction
(×0.25 after 10 epochs without improvement) and early stopping (patience 25, best weights restored),
both on validation loss.

This notebook trains the models once for every input normalization option (and, optionally, with and
without overlapping training subcubes), shows a comparison table, and publishes the configuration you
choose. Use a GPU runtime (Runtime → Change runtime type → GPU). One configuration took 2.7 minutes
on a Colab GPU without overlap; with overlap, epochs are about 3.6× longer. The last cell downloads
`deepvoid_demo_outputs.zip`.
"""),
    code(f'''
import os, subprocess
os.environ["GIT_TERMINAL_PROMPT"] = "0"     # fail fast instead of asking for a password
if not os.path.isdir("dvdemo"):
    if os.path.isdir("../dvdemo"):
        os.chdir("..")
    elif subprocess.run(["git", "clone", "-q", "--depth", "1",
                         "https://github.com/{REPO}.git"]).returncode == 0:
        os.chdir("deepvoid-demo")
    else:  # repo not public yet: upload the zip of the repo instead
        from google.colab import files
        print("Upload deepvoid-demo.zip")
        name = next(iter(files.upload()))
        subprocess.run(["unzip", "-q", "-o", name], check=True)
        os.chdir("deepvoid-demo")
print("working in", os.getcwd())
'''),
    code('''
import json, shutil, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from dvdemo.data import Normalizer, split_masks
from dvdemo.io import load_sample
from dvdemo.train import ArchConfig, StageConfig, run_all

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| GPUs:", gpus)
if not gpus:
    print("WARNING: no GPU - training will be very slow. Switch the runtime to GPU.")
sample = load_sample("data/tng300_128.npz")
'''),
    md("""
## Choices

Two settings are open choices rather than fixed parts of the method. Both are saved in each model's
`.json`.

### 1. Input normalization

The stored fields are ρ/ρ̄ = 1 + δ. All three options are fitted on the
training voxels only and reused unchanged at prediction time.

| Option | Transform | Rationale |
|---|---|---|
| `"minmax"` | (x − min) / (max − min) | What the DeepVoid paper uses. Simple, bounded to [0, 1], no free parameters, and identical whether the input is ρ, ρ/ρ̄ or δ. |
| `"log"` | log10(1 + δ), then min-max | Cosmological density fields are close to log-normal, so the log spreads out the low-density range where voids and walls differ, and puts dense and sparse tracers on a common relative scale. Empty voxels (93% of the galaxy field) are floored at ρ/ρ̄ = 10⁻³ before the log. |
| `"log_zscore"` | log10(1 + δ), then (y − mean) / std | Same log; scaling by the mean and spread of the distribution rather than its two most extreme voxels, so a single unusually dense halo does not change the scale. |

### 2. Overlapping training subcubes

Without overlap, training uses the 51 non-overlapping training
subcubes (faster). With overlap it adds subcubes at a stride of 16 voxels
that lie entirely inside training subcubes (185 in total): more varied training data, longer epochs.
Validation subcubes are never touched either way.

The histograms show what each normalization does to the training voxels of both input fields.
"""),
    code('''
train_mask, _ = split_masks(sample["mask"].shape, sample["meta"]["split"])
fig, axes = plt.subplots(2, 3, figsize=(12, 5.4))
for r, (field, title) in enumerate((("dm", "Dark matter, λ = 0.33 h⁻¹ Mpc"),
                                    ("subhalo_L7", "Galaxies, λ = 7 h⁻¹ Mpc"))):
    x = sample[field][train_mask]
    for c, method in enumerate(("minmax", "log", "log_zscore")):
        v = Normalizer(method).fit(x).transform(x)
        axes[r, c].hist(v, bins=100, color="#2f5d8a", log=True)
        axes[r, c].set_title(f"{method}: {title}", fontsize=9)
        axes[r, c].spines[["top", "right"]].set_visible(False)
for a in axes[1]: a.set_xlabel("model input value")
for a in axes[:, 0]: a.set_ylabel("voxels (log scale)")
plt.tight_layout(); plt.show()
'''),
    md("Smoke test (a few minutes): runs every stage for one epoch and writes to `/tmp`, to catch problems before the long run."),
    code('''
tiny = dict(max_epochs=1)
_ = run_all(sample, out_dir="/tmp/smoke/weights", results_dir="/tmp/smoke/results",
            dense=StageConfig(**tiny), curriculum=StageConfig(**tiny, learning_rate=1e-4, frozen_blocks=1),
            scratch=StageConfig(**tiny))
print("smoke test passed")
'''),
    md("""
## Train every configuration

Each configuration trains all three models from scratch and writes them to `runs/<configuration>/`.
Edit the two lists to run fewer.
"""),
    code('''
NORMS = ["minmax", "log", "log_zscore"]
OVERLAPS = [False, True]
ARCH = ArchConfig(depth=3, filters=16, batch_norm=True, dropout=0.0)

rows = []
for overlap in OVERLAPS:
    for norm in NORMS:
        tag = f"{norm}{'_overlap' if overlap else ''}"
        common = dict(max_epochs=300, batch_size=8, normalization=norm, overlap=overlap, verbose=0)
        t = time.time()
        print(f"=== {tag} ===", flush=True)
        res = run_all(sample, out_dir=f"runs/{tag}/weights", results_dir=f"runs/{tag}/results", arch=ARCH,
                      dense=StageConfig(learning_rate=1e-3, seed=0, **common),
                      curriculum=StageConfig(learning_rate=1e-4, frozen_blocks=1, seed=1, **common),
                      scratch=StageConfig(learning_rate=1e-3, seed=2, **common))
        print(f"    {(time.time() - t) / 60:.1f} min", flush=True)
        for name, r in res.items():
            m = r["val_metrics"]
            rows.append({"config": tag, "model": name, "void_f1": m["void_f1"], "mcc": m["mcc"],
                         "balanced_acc": m["balanced_accuracy"], "halo_f1": m["halo_f1"]})

summary = pd.DataFrame(rows)
summary.to_csv("runs/summary.csv", index=False)
summary.pivot(index="model", columns="config", values="mcc").round(3)
'''),
    md("""
## Choose the configuration to publish

The table above shows validation MCC; the next cell shows every metric. By default the cell picks the
configuration with the highest mean validation MCC across the three trained models; set `CHOSEN`
yourself to override. Note that picking a configuration on the validation subcubes (which also drive
early stopping) makes its validation scores slightly optimistic, which the README mentions.
"""),
    code('''
trained = summary[summary.model != "subhalo_no_retraining"]
ranking = trained.groupby("config").mcc.mean().sort_values(ascending=False)
print("mean validation MCC of the three trained models:")
print(ranking.round(4).to_string())

CHOSEN = ranking.index[0]          # or e.g. CHOSEN = "minmax"
print("\\npublishing:", CHOSEN)
for d in ("weights", "results"):
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(f"runs/{CHOSEN}/{d}", d)
with open("results/chosen_config.json", "w") as f:
    json.dump({"chosen": CHOSEN, "rule": "highest mean validation MCC of the three trained models",
               "mean_mcc": ranking.to_dict()}, f, indent=2)
shutil.copy("runs/summary.csv", "results/all_configs.csv")
summary[summary.config == CHOSEN].set_index("model").round(3)
'''),
    code('''
!python scripts/make_figures.py
from IPython.display import Image, display
display(Image("assets/hero.gif"))
'''),
    code('''
!zip -q -r deepvoid_demo_outputs.zip weights results assets/hero.gif assets/*.png runs/*/results/metrics.json runs/summary.csv
try:
    from google.colab import files
    files.download("deepvoid_demo_outputs.zip")
except ImportError:
    print("wrote deepvoid_demo_outputs.zip")
'''),
]

SEEDS = [
    md(f"""
# Seed study: how much do the scores vary between training runs?

{badge("notebooks/seed_study.ipynb")}

This notebook is optional. You don't need it to use the demo; the README already shows its result
(`assets/seed_study.png`). It answers one question: are the differences between models, especially
curriculum vs. training from scratch, bigger than the run-to-run noise?

It retrains the three models several times with the published settings (min-max normalization,
overlapping training subcubes) and the same training/validation split, changing only the training
seed (weight initialization and shuffling). Each seed takes about 4 minutes on a Colab GPU; the default
5 seeds take about 20 minutes. Use a GPU runtime. The last cell downloads `seed_study_outputs.zip`.
"""),
    TRAIN[1],
    code('''
import json, os, time
import numpy as np
import tensorflow as tf
from dvdemo.io import load_sample
from dvdemo.train import ArchConfig, StageConfig, run_all

print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
sample = load_sample("data/tng300_128.npz")
'''),
    code('''
N_SEEDS = 5
common = dict(max_epochs=300, batch_size=8, normalization="minmax", overlap=True, verbose=0)
runs = []
for k in range(N_SEEDS):
    t = time.time()
    print(f"=== seed {k} ===", flush=True)
    res = run_all(sample, out_dir=f"runs_seeds/seed{k}/weights", results_dir=f"runs_seeds/seed{k}/results",
                  arch=ArchConfig(),
                  dense=StageConfig(learning_rate=1e-3, seed=100 * k, **common),
                  curriculum=StageConfig(learning_rate=1e-4, frozen_blocks=1, seed=100 * k + 1, **common),
                  scratch=StageConfig(learning_rate=1e-3, seed=100 * k + 2, **common))
    runs.append({name: {m: r["val_metrics"][m] for m in ("void_f1", "mcc", "balanced_accuracy", "halo_f1")}
                 for name, r in res.items()})
    print(f"    {(time.time() - t) / 60:.1f} min", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/seed_study.json", "w") as f:
    json.dump({"config": "minmax + overlap", "split_seed": 12, "training_seeds": N_SEEDS,
               "runs": runs}, f, indent=2)
'''),
    code('''
!python scripts/make_seed_figure.py
from IPython.display import Image, display
display(Image("assets/seed_study.png"))
'''),
    code('''
!zip -q seed_study_outputs.zip results/seed_study.json assets/seed_study.png
try:
    from google.colab import files
    files.download("seed_study_outputs.zip")
except ImportError:
    print("wrote seed_study_outputs.zip")
'''),
]

def main():
    os.makedirs("notebooks", exist_ok=True)
    for name, cells, gpu in (("deepvoid_demo", DEMO, False), ("train_demo_models", TRAIN, True),
                              ("seed_study", SEEDS, True)):
        path = f"notebooks/{name}.ipynb"
        with open(path, "w") as f:
            json.dump(notebook(cells, gpu), f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
