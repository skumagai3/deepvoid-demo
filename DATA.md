# Data card: `data/tng300_128.npz`

This file is a 128³ sample of the IllustrisTNG TNG300-3-Dark simulation at redshift zero. It's small enough
to keep in the repo (4.5 MB) and large enough to train and evaluate the demo models. It was made with
[`scripts/prepare_sample.py`](scripts/prepare_sample.py) from three `.fvol` volumes I produced during the
DeepVoid project in 2024.

TNG300-3-Dark is the dark-matter-only, lowest-resolution run of the TNG300 box (205 h⁻¹ Mpc on a side,
625³ particles). DeepVoid only needs the dark matter particles and the subhalos identified in them, and the
lower resolution makes training and processing much cheaper than the full-physics or higher-resolution runs.

| Key | dtype | Contents |
|---|---|---|
| `dm` | float16 | dark matter density on a 128³ grid, as ρ/ρ̄. Mean particle spacing λ = 0.33 h⁻¹ Mpc. |
| `subhalo_L7` | float16 | subhalo mass density (subhalos are the clumps of dark matter where galaxies would form, so they stand in for galaxies), with subhalos subsampled to a mean separation of λ = 7 h⁻¹ Mpc, as ρ/ρ̄. 92.6% of voxels are empty. |
| `mask` | uint8 | T-web class for each voxel: 0 void, 1 wall, 2 filament, 3 halo. |
| `meta` | JSON string | box size, grid, label settings, class fractions, the training/validation split, and SHA-256 hashes of the source files. |

`dvdemo.io.load_sample` returns the fields as float32 and `meta` as a dict.

## Details

Geometry: a periodic box of 205 h⁻¹ Mpc with 128³ voxels of 1.6 h⁻¹ Mpc. All three arrays use the same
grid and axis order.

Labels: the T-web classification (the number of tidal tensor eigenvalues above 0.65) of the full dark matter
field, with the potential smoothed by a Gaussian of σ = 0.6 grid cells (about 1 h⁻¹ Mpc). That is the 128³
equivalent of the σ = 2.4 cells used on the 512³ grid in the paper. The README explains how the tidal
tensor is computed. The class fractions are 61.6% void, 32.9% wall, 5.0% filament and 0.5% halo. Both inputs are
trained against these same labels, so with the subhalo input the network has to infer structure in matter
it can't see.

Split: the box is cut into 64 subcubes of 32³, which are split at random into 51 training and 13 validation
subcubes (seed 12). The split is stored in `meta["split"]` as subcube indices (i, j, k), where each index
covers voxels 32i to 32i + 31 along that axis. Only training subcubes are augmented (four copies: the
original and 90° rotations about each axis), normalization statistics are computed from training voxels
only, and all metrics are computed on the validation subcubes.

## Checks

Since these volumes were made a while ago, I checked them before using them:

- [`assets/alignment.png`](assets/alignment.png) shows slices through x, y and z of all three fields, with
  the mask's void boundaries drawn over both inputs. The fields line up.
- Median dark matter density increases from void to wall to filament to halo (about 0.23, 0.62, 3.3 and
  23 times the mean), as it should.

## Credit and citation

These data are derived from the public IllustrisTNG release produced by the IllustrisTNG collaboration
([tng-project.org](https://www.tng-project.org/)). Please use them under the
[TNG data access terms](https://www.tng-project.org/data/) and cite the IllustrisTNG introductory papers and
the data release paper:

- Springel, V., et al. 2018, MNRAS, 475, 676 ([doi:10.1093/mnras/stx3304](https://doi.org/10.1093/mnras/stx3304))
- Pillepich, A., et al. 2018, MNRAS, 475, 648 ([doi:10.1093/mnras/stx3112](https://doi.org/10.1093/mnras/stx3112))
- Nelson, D., et al. 2018, MNRAS, 475, 624 ([doi:10.1093/mnras/stx3040](https://doi.org/10.1093/mnras/stx3040))
- Naiman, J. P., et al. 2018, MNRAS, 477, 1206 ([doi:10.1093/mnras/sty618](https://doi.org/10.1093/mnras/sty618))
- Marinacci, F., et al. 2018, MNRAS, 480, 5113 ([doi:10.1093/mnras/sty2206](https://doi.org/10.1093/mnras/sty2206))
- Nelson, D., et al. 2019, Computational Astrophysics and Cosmology, 6, 2 ([doi:10.1186/s40668-019-0028-x](https://doi.org/10.1186/s40668-019-0028-x)), the public data release

If you use the labels or the models, please also cite the [DeepVoid paper](https://doi.org/10.3847/1538-4357/ae2c80).
