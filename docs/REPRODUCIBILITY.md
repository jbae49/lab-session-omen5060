# Reproducibility manifest

## Repository revisions

| component | upstream base | lab snapshot commit |
|---|---|---|
| PromptHMR | `3b566b7dbb28ce506c7ea972c18693f4c705ce8c` | `218b6c9b496bab1bf119bf66ada42574355e546d` |
| GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` | `54e1ee563da7ea920cdd08d6975fc88e058c5d21` |
| Unitree RL MjLab | local base `05c7538e95a300edf7dfdd5fca5907f0bf00cd43` | `8cbbbeadfaa2499e452e6da5356947e0b06244f0` |

The table records the source snapshots from which this teaching bundle was prepared.
Nested `.git` histories are deliberately excluded so the class repository stays small;
the source needed by the lab is vendored directly in `repos/`.

## Lab-specific source changes

- PromptHMR: click/rank subject initialization and forward/backward SAM2 propagation.
- GMR: robust SMPL-X beta count/expression handling.
- Unitree RL MjLab: uniquely named 29DoF class tasks and the minimal G1 asset snapshot.

## OMEN 16 / RTX 5060 classroom environment

- Python 3.11 in all three environments
- PyTorch `2.7.1`, torchvision `0.22.1`, CUDA wheel `cu128`
- xFormers `0.0.31.post1`
- Detectron2 source revision `247691a1c292654867deb35cabb27f18d75a1363`, built for `sm_120`
- SAM2 source revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`, optional CUDA postprocessing disabled
- MJLab `1.2.0`, MuJoCo-Warp `3.5.0`
- MuJoCo `3.5.0`, Warp-Lang `1.12.0` (matching the MJLab v1.2.0 lock)
- PromptHMR input height 640, segmentation batch 2, image-model batch 1
- PPO profiles: 32-env smoke, 128-env quick, 256-env full

The optional after-class workflow uses the same pinned package versions but separate Conda
environments (`phmr_postclass_omen5060`, `gmr_postclass_omen5060`, and
`mjlab_postclass_omen5060`) and a separate RL experiment root (`postclass_g1_tracking`).
It does not mutate or activate the classroom environments.

The top-level teaching repository is the versioned unit. Do not run Git commands inside
the three vendored source directories.

Deterministic bit-for-bit GPU learning is not guaranteed by PPO/MuJoCo-Warp, but code,
assets, seed, task, reference, and hyperparameters are fixed here for experimental
reproducibility.
