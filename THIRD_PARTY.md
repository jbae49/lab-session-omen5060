# Third-party source snapshots

This teaching repository vendors a minimal, modified source snapshot of three upstream
projects so students can run one reproducible pipeline without nested Git repositories.
Their original licenses remain in each directory and continue to apply.

| Component | Upstream | Recorded base revision | Local purpose |
|---|---|---|---|
| PromptHMR | https://github.com/yufu-wang/PromptHMR | `3b566b7dbb28ce506c7ea972c18693f4c705ce8c` | RGB video → SMPL-X |
| GMR | https://github.com/YanjieZe/GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` | SMPL-X → G1 retargeting |
| Unitree RL MjLab | https://github.com/unitreerobotics/unitree_rl_mjlab | local lineage documented in `docs/REPRODUCIBILITY.md` | reference conversion, PPO, rendering |

Lab-specific changes include deterministic subject selection, bidirectional tracking,
RTX 5060 memory settings, headless visualization, G1 grounding, and motion-imitation task
configuration. The vendored directories intentionally contain no nested `.git` histories.

SMPL and SMPL-X model files are not redistributed. Each student must accept the official
licenses and download them from the links in the main README.
