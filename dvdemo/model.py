"""A compact 3D U-Net in Keras.

Encoder blocks halve the resolution and double the filters; the decoder
upsamples and concatenates the matching encoder features (skip connections),
which keeps fine spatial detail. The network is fully convolutional, so it
accepts any input side divisible by ``2 ** depth``. Only standard Keras
layers are used, so saved ``.keras`` files load without custom objects.
"""
from __future__ import annotations

from typing import List


def _conv_block(x, filters: int, name: str, batch_norm: bool, dropout: float):
    from tensorflow.keras import layers as L
    for i in range(2):
        x = L.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal",
                     use_bias=not batch_norm, name=f"{name}_conv{i}")(x)
        if batch_norm:
            x = L.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = L.Activation("relu", name=f"{name}_relu{i}")(x)
    if dropout:
        x = L.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def build_unet(depth: int = 3, filters: int = 16, num_classes: int = 4,
               batch_norm: bool = True, dropout: float = 0.0, name: str = "deepvoid_unet"):
    """Return an uncompiled 3D U-Net mapping (B, N, N, N, 1) -> (B, N, N, N, classes)."""
    import tensorflow as tf
    from tensorflow.keras import layers as L

    inputs = L.Input((None, None, None, 1), name="density")
    x, skips = inputs, []
    for d in range(depth):
        x = _conv_block(x, filters * 2 ** d, f"enc{d}", batch_norm, dropout)
        skips.append(x)
        x = L.MaxPooling3D(2, name=f"enc{d}_pool")(x)
    x = _conv_block(x, filters * 2 ** depth, "bottleneck", batch_norm, dropout)
    for d in reversed(range(depth)):
        x = L.Conv3DTranspose(filters * 2 ** d, 2, strides=2, name=f"dec{d}_up")(x)
        x = L.Concatenate(name=f"dec{d}_concat")([x, skips[d]])
        x = _conv_block(x, filters * 2 ** d, f"dec{d}", batch_norm, dropout)
    x = L.Conv3D(num_classes, 1, name="logits")(x)
    out = L.Activation("softmax", dtype="float32", name="probs")(x)
    return tf.keras.Model(inputs, out, name=name)


def freeze_encoder(model, n_blocks: int) -> List[str]:
    """Freeze encoder blocks enc0..enc{n-1} (convs and their BatchNorm, which
    then also runs in inference mode); everything else stays trainable.
    Recompile the model afterwards. Returns the names of frozen layers."""
    prefixes = tuple(f"enc{i}_" for i in range(n_blocks))
    frozen = []
    for layer in model.layers:
        layer.trainable = not (prefixes and layer.name.startswith(prefixes))
        if not layer.trainable and layer.weights:
            frozen.append(layer.name)
    return frozen


def count_params(model) -> int:
    return int(model.count_params())
