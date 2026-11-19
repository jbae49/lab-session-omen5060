# Pipeline input/output contract

| Stage | Input | Output | Critical convention |
|---|---|---|---|
| Class input | baked joint-target NPZ | positions `(T,14,3)`, quaternions `(T,14,4)` | quaternion is wxyz; no body model needed |
| PromptHMR | RGB frames `(H,W,3)` | per-track SMPL-X pose/shape/trans | world is Y-up |
| Export | `results.pkl`, track id | SMPL-X NPZ | rotate to Z-up |
| GMR | SMPL-X NPZ | `qpos (T,36)` | root quaternion is wxyz; 29 joints |
| MjLab preprocess | G1 qpos NPZ | full 50 Hz reference NPZ | MuJoCo FK body states |
| PPO | reference NPZ | `.pt`, `.onnx` | actor action dimension 29 |

## Coordinate/frame notation

- `_w`: world frame
- `_b`: current robot base/pelvis frame
- `q`: generalized configuration; free root plus joint angles
- `p`: position vector
- orientation is stored as quaternion or 6D rotation representation, never Euler angles

## Validation invariants

```text
PromptHMR selected track count == intended people
class joint-target format == gmr_joint_targets_v1
SMPL-X frames == selected track frames
GMR qpos width == 36
MjLab joint_pos width == 29
MjLab body arrays share the same T
all numeric arrays finite
reference fps == 50 for the supplied tasks
policy action width == 29
```
