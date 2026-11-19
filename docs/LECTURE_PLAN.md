# Suggested lecture plan

## Before class (retarget-only default)

- Clone the repository and run `scripts/install_retarget_lab.sh --execute`.
- Verify `sample_data/input/jump_smplx_joint_targets_30fps.npz`.
- Run `scripts/run_retarget_lab.sh` once and play the resulting MP4.
- PromptHMR and licensed SMPL/SMPL-X files are not required in class.

## 120-minute class

1. 0–15 min: SMPL/SMPL-X concept and why motion reconstruction is precomputed.
2. 15–30 min: inspect the supplied joint-target NPZ and its coordinate conventions.
3. 30–55 min: GMR IK objective, robot DoF mismatch, link targets and joint limits.
4. 55–75 min: run SMPL-X joints → G1 29DoF retargeting.
5. 75–90 min: inspect qpos, root quaternion, grounding and numerical validity.
6. 90–105 min: convert qpos to a 50Hz MjLab reference and inspect FK body states.
7. 105–112 min: play the recorded G1 MP4 and discuss retargeting errors.
8. 112–117 min: run the 20-iteration RL smoke test.
9. 117–120 min: start the 60k background run and verify that it is alive.

```bash
bash scripts/train_mimic.sh smoke
bash scripts/start_training.sh full
bash scripts/training_status.sh
```

The long run is intentionally not expected to finish during class. Keep the laptop on AC
power, prevent suspend, and inspect the newest checkpoint and rollout the next day:

```bash
bash scripts/training_status.sh
bash scripts/record_latest_policy.sh
```

## Assignment

Students modify or compare a supplied person joint motion, produce a valid 29DoF reference,
and submit:

- supplied/modified joint-target NPZ
- GMR/MjLab validation report
- G1 retarget MP4
- training log, newest checkpoint iteration, and next-day policy rollout MP4
- a short analysis of the largest retargeting error
