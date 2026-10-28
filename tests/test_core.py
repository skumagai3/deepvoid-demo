"""Tests for the parts of the pipeline that do not need TensorFlow."""
import numpy as np
import pytest

from dvdemo.data import (Normalizer, build_arrays, cell_mask, check_split, make_split,
                         rotations, split_masks, training_corners)
from dvdemo.inference import predict_volume, tile_plan
from dvdemo.io import read_fvol, read_mask, write_fvol
from dvdemo.metrics import evaluate


def test_fvol_roundtrip(tmp_path):
    vol = np.random.default_rng(0).random((8, 6, 4)).astype(np.float32)
    write_fvol(tmp_path / "a.fvol", vol)
    np.testing.assert_array_equal(read_fvol(str(tmp_path / "a.fvol")), vol)


def test_read_mask_rejects_bad_values(tmp_path):
    write_fvol(tmp_path / "m.fvol", np.full((4, 4, 4), 2.0))
    assert read_mask(str(tmp_path / "m.fvol")).max() == 2
    write_fvol(tmp_path / "bad.fvol", np.full((4, 4, 4), 5.0))
    with pytest.raises(ValueError):
        read_mask(str(tmp_path / "bad.fvol"))


def test_tile_plan_covers_axis_once():
    for n, size, stride in [(128, 64, 32), (64, 32, 16), (128, 96, 64)]:
        keeps = [k for _, k in tile_plan(n, size, stride)]
        covered = np.concatenate([np.arange(k, k + stride) for k in keeps])
        assert sorted(covered) == list(range(n))


def test_predict_volume_reassembles_exactly():
    """With a 'model' that only looks at each voxel, stitching must reproduce
    the voxelwise answer everywhere, including across the periodic faces."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=(64, 64, 64))
    x = labels.astype(np.float32)

    def pointwise(batch):
        return np.eye(4, dtype=np.float32)[batch[..., 0].astype(int)]

    probs = predict_volume(pointwise, x, size=32, stride=16, batch_size=5)
    np.testing.assert_array_equal(probs.argmax(-1), labels)


def test_predict_volume_uses_context_from_wrapped_neighbours():
    """A model whose output depends on neighbours: the stitched result must
    equal applying the same operation to the whole periodic box."""
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(1)
    x = rng.random((32, 32, 32)).astype(np.float32)

    def smooth(batch):  # 3x3x3 box mean; context of 1 voxel is enough
        out = np.stack([uniform_filter(b[..., 0], 3, mode="nearest") for b in batch])
        return np.stack([out, -out], axis=-1)

    probs = predict_volume(smooth, x, size=16, stride=8, num_classes=2)
    np.testing.assert_allclose(probs[..., 0], uniform_filter(x, 3, mode="wrap"), rtol=1e-5)


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    t = rng.integers(0, 4, 5000)
    p = np.where(rng.random(5000) < 0.7, t, rng.integers(0, 4, 5000))
    s = evaluate(t, p, ["void", "wall", "filament", "halo"])
    assert abs(s["mcc"] - sk.matthews_corrcoef(t, p)) < 1e-12
    assert abs(s["void_f1"] - sk.f1_score(t, p, labels=[0], average=None)[0]) < 1e-12
    assert abs(s["balanced_accuracy"] - sk.balanced_accuracy_score(t, p)) < 1e-12


def test_split_is_disjoint_and_covers_the_box():
    sp = make_split(128, 32)
    check_split(sp, 128)
    assert len(sp["val"]) == 13 and len(sp["train"]) == 51
    train, val = split_masks((128,) * 3, sp)
    assert not (train & val).any() and (train | val).all()
    assert make_split(128, 32) == sp                       # reproducible (seed 12)
    with pytest.raises(ValueError):
        check_split(dict(sp, val=sp["val"] + sp["train"][:1]), 128)


def test_only_training_subcubes_are_augmented_and_validation_stays_out():
    """Label each voxel with the id of its subcube: training arrays may only
    contain training ids (also with overlap), validation arrays only validation
    ids, and only the training arrays get the 4 rotations."""
    n, s = 128, 32
    sp = make_split(n, s)
    ids = np.zeros((n,) * 3, np.int32)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ids[i*s:(i+1)*s, j*s:(j+1)*s, k*s:(k+1)*s] = 16 * i + 4 * j + k
    train_ids = {16 * i + 4 * j + k for i, j, k in sp["train"]}
    val_ids = {16 * i + 4 * j + k for i, j, k in sp["val"]}
    for overlap in (False, True):
        xt, yt, xv, yv = build_arrays(ids.astype(np.float32), ids, sp, overlap=overlap)
        assert set(np.unique(yt)) <= train_ids
        assert set(np.unique(yv)) == val_ids
        assert len(xt) == 4 * len(training_corners(sp, n, overlap)) and len(xv) == 13
    assert len(training_corners(sp, n, overlap=True)) > len(sp["train"])


def test_rotations_keep_input_and_label_aligned():
    rng = np.random.default_rng(3)
    a = rng.random((8, 8, 8))
    ra, rb = rotations(a), rotations(a.copy())
    assert len(ra) == 4
    for x, y in zip(ra, rb):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(np.sort(x.ravel()), np.sort(a.ravel()))
    assert all(not np.array_equal(r, a) for r in ra[1:])   # each rotation is a real change


def test_normalizer_options():
    from dvdemo.data import LOG_FLOOR
    rng = np.random.default_rng(0)
    train = rng.lognormal(sigma=2, size=(32, 32, 32))      # stands in for rho / rho_bar
    other = 10 * rng.lognormal(size=(8, 8, 8))
    mm = Normalizer("minmax").fit(train)
    t = mm.transform(train)
    assert abs(t.min()) < 1e-6 and abs(t.max() - 1) < 1e-6
    np.testing.assert_allclose(mm.transform(other),            # not refitted on new data
                               (other - train.min()) / (train.max() - train.min()), rtol=1e-5)
    lg = Normalizer("log").fit(train)
    t = lg.transform(train)
    assert abs(t.min()) < 1e-6 and abs(t.max() - 1) < 1e-6
    ly = np.log10(np.maximum(train, LOG_FLOOR))               # log10(1 + delta), floored
    np.testing.assert_allclose(t, (ly - ly.min()) / (ly.max() - ly.min()), rtol=1e-4, atol=1e-6)
    lz = Normalizer("log_zscore").fit(train)
    t = lz.transform(train)
    assert abs(t.mean()) < 1e-5 and abs(t.std() - 1) < 1e-4
    sparse = np.zeros((16, 16, 16)); sparse[::4, ::4, ::4] = 5.0   # empty voxels hit the floor
    for m in ("log", "log_zscore"):
        n = Normalizer(m).fit(sparse)
        out = n.transform(sparse)
        assert np.isfinite(out).all() and n.stats["floor"] == LOG_FLOOR
        assert len(np.unique(out)) == 2
    for norm in (mm, lg, lz):
        back = Normalizer.from_dict(norm.to_dict())
        assert back.method == norm.method and back.stats == norm.stats
    with pytest.raises(ValueError):
        Normalizer("zscore")
