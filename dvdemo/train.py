"""Training for the demo models: fit on the
(augmented) training subcubes, monitor validation loss for learning-rate
reduction and early stopping, restore the best weights.

Three models are trained, plus one comparison without retraining:

* ``dm_model``               trained on the dark-matter field;
* ``subhalo_curriculum``     ``dm_model`` fine-tuned on the lambda = 7 subhalo field,
                             first encoder block frozen (one-step curriculum learning);
* ``subhalo_from_scratch``   a fresh model on the subhalo field, same settings;
* ``subhalo_no_retraining``  ``dm_model`` applied to the subhalo field unchanged.

Every model is scored on the voxels of the validation subcubes, using a
full-box prediction (so each validation voxel sees its real surroundings as
input). Each run writes ``<name>.keras`` and ``<name>.json`` (normalization,
architecture, split, settings, history and metrics).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from .data import Normalizer, build_arrays, split_masks
from .inference import keras_predict_fn, predict_volume
from .io import CLASS_NAMES
from .metrics import evaluate


@dataclass
class StageConfig:
    max_epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 1e-3
    lr_patience: int = 10        # ReduceLROnPlateau(factor=0.25)
    patience: int = 25           # EarlyStopping(restore_best_weights=True)
    frozen_blocks: int = 0
    normalization: str = "minmax"
    overlap: bool = False
    seed: int = 0
    verbose: int = 2             # Keras fit verbosity: 2 = one line per epoch, 0 = silent


@dataclass
class ArchConfig:
    depth: int = 3
    filters: int = 16
    batch_norm: bool = True
    dropout: float = 0.0


def score(model, x_norm, y, where, size: int = 32, stride: int = 16):
    """Predict the full box and score the voxels in ``where``; returns (metrics, labels)."""
    probs = predict_volume(keras_predict_fn(model), x_norm, size=size, stride=stride)
    pred = probs.argmax(-1).astype(np.uint8)
    return evaluate(y, pred, list(CLASS_NAMES), where=where), pred


def train_stage(model, x_raw: np.ndarray, y: np.ndarray, split: Dict, cfg: StageConfig):
    """Fit ``model`` on the training subcubes of ``x_raw`` -> ``y``.

    Returns (normalizer fitted on the training voxels, info dict with history)."""
    import tensorflow as tf
    from .model import freeze_encoder

    tf.keras.utils.set_random_seed(cfg.seed)
    train_mask, _ = split_masks(y.shape, split)
    norm = Normalizer(cfg.normalization).fit(x_raw[train_mask])
    xt, yt, xv, yv = build_arrays(norm.transform(x_raw), y, split, overlap=cfg.overlap)
    frozen = freeze_encoder(model, cfg.frozen_blocks)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.25,
                                             patience=cfg.lr_patience, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                         restore_best_weights=True),
    ]
    t0 = time.time()
    hist = model.fit(xt, yt, validation_data=(xv, yv), batch_size=cfg.batch_size,
                     epochs=cfg.max_epochs, shuffle=True, callbacks=callbacks, verbose=cfg.verbose)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    info = {"seconds": round(time.time() - t0, 1), "epochs_run": len(h["loss"]),
            "best_epoch": int(np.argmin(h["val_loss"])) + 1,
            "n_train_samples": int(len(xt)), "n_val_subcubes": int(len(xv)),
            "frozen_layers": frozen, "history": h}
    return norm, info


def save_run(model, path_stem: str, norm: Normalizer, record: dict) -> None:
    import tensorflow as tf
    os.makedirs(os.path.dirname(path_stem) or ".", exist_ok=True)
    # Save an uncompiled, fully trainable copy: smaller file (no optimizer state).
    # (Keras 3's clone_model copies the compile state, so rebuild from the config.)
    clean = tf.keras.Model.from_config(model.get_config())
    clean.set_weights(model.get_weights())
    for layer in clean.layers:
        layer.trainable = True
    clean.save(path_stem + ".keras")
    with open(path_stem + ".json", "w") as f:
        json.dump(dict(record, normalizer=norm.to_dict()), f, indent=2, default=float)


def load_run(path_stem: str):
    """Load a saved model and its record; returns (model, normalizer, record)."""
    import tensorflow as tf
    with open(path_stem + ".json") as f:
        record = json.load(f)
    model = tf.keras.models.load_model(path_stem + ".keras", compile=False)
    return model, Normalizer.from_dict(record["normalizer"]), record


def run_all(sample: dict, out_dir: str = "weights", arch: Optional[ArchConfig] = None,
            dense: Optional[StageConfig] = None, curriculum: Optional[StageConfig] = None,
            scratch: Optional[StageConfig] = None, results_dir: str = "results") -> Dict[str, dict]:
    """Train the three demo models and score everything on the validation subcubes."""
    import tensorflow as tf
    from .model import build_unet

    arch = arch or ArchConfig()
    dense = dense or StageConfig()
    curriculum = curriculum or StageConfig(learning_rate=1e-4, frozen_blocks=1, seed=1)
    scratch = scratch or StageConfig(seed=2)
    split = sample["meta"]["split"]
    y = sample["mask"]
    _, val_mask = split_masks(y.shape, split)
    env = {"tensorflow": tf.__version__,
           "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}
    results, preds = {}, {}

    def finish(name, model, norm, info, field, stages):
        s, pred = score(model, norm.transform(sample[field]), y, val_mask)
        preds[name] = pred
        rec = {"name": name, "input_field": field,
               "lambda_mpc_h": sample["meta"]["fields"][field]["lambda_mpc_h"],
               "arch": asdict(arch), "stages": stages, "split": split, "env": env,
               "params": int(model.count_params()), "train": info, "val_metrics": s}
        save_run(model, os.path.join(out_dir, name), norm, rec)
        results[name] = rec
        print(f"{name}: validation MCC {s['mcc']:.4f}, void F1 {s['void_f1']:.4f}", flush=True)

    print("== 1/3 dense dark matter (lambda = 0.33) ==", flush=True)
    m = build_unet(**asdict(arch))
    norm, info = train_stage(m, sample["dm"], y, split, dense)
    finish("dm_model", m, norm, info, "dm", [asdict(dense)])

    # Zero-shot: the dense model on the sparse field, no retraining
    # (input scaling refitted on the sparse field's training voxels).
    train_mask, _ = split_masks(y.shape, split)
    zs = Normalizer(dense.normalization).fit(sample["subhalo_L7"][train_mask])
    s, preds["subhalo_no_retraining"] = score(m, zs.transform(sample["subhalo_L7"]), y, val_mask)
    results["subhalo_no_retraining"] = {"name": "subhalo_no_retraining", "val_metrics": s,
                              "note": "dm_model weights applied to the L=7 subhalo field"}
    print(f"subhalo_no_retraining: validation MCC {s['mcc']:.4f}, void F1 {s['void_f1']:.4f}", flush=True)

    print("== 2/3 curriculum: dense model fine-tuned on galaxies (lambda = 7) ==", flush=True)
    m2 = tf.keras.models.clone_model(m)
    m2.set_weights(m.get_weights())
    norm2, info2 = train_stage(m2, sample["subhalo_L7"], y, split, curriculum)
    finish("subhalo_curriculum", m2, norm2, info2, "subhalo_L7", [asdict(dense), asdict(curriculum)])

    print("== 3/3 galaxies from scratch ==", flush=True)
    m3 = build_unet(**asdict(arch))
    norm3, info3 = train_stage(m3, sample["subhalo_L7"], y, split, scratch)
    finish("subhalo_from_scratch", m3, norm3, info3, "subhalo_L7", [asdict(scratch)])

    os.makedirs(results_dir, exist_ok=True)
    np.savez_compressed(os.path.join(results_dir, "predictions.npz"), **preds)
    keys = ("mcc", "void_f1", "balanced_accuracy", "f1_macro", "accuracy",
            "wall_f1", "filament_f1", "halo_f1", "n_voxels")
    summary = {k: {kk: v["val_metrics"][kk] for kk in keys} for k, v in results.items()}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({"split": split, "env": env, "validation": summary}, f, indent=2)
    return results
