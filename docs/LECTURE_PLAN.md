# Suggested lecture plan

## Before class

- Install all three Conda environments.
- Place licensed SMPL-X models and run all `doctor.py` checks.
- Run the full video-to-reference pipeline once.
- Verify the one-command smoke checkpoint and recorded rollout.
- Keep the expected intermediate files; they let students continue if one GPU stage fails.

## 120-minute class

1. 0–15 min: pose, shape, root transform; why a mesh model is useful.
2. 15–30 min: SMPL versus SMPL-X; inspect the supplied overlay.
3. 30–45 min: detector, click/rank person selection, SAM2 temporal tracking and occlusion.
4. 45–60 min: coordinate conversion and inspect the exported SMPL-X arrays.
5. 60–80 min: GMR IK objective, robot DoF mismatch, joint limits and qpos.
6. 80–95 min: convert qpos to MjLab reference and inspect FK body states.
7. 95–110 min: PPO observation/action/reward/termination/DR/RSI.
8. 110–120 min: play the smoke policy recording and compare reference versus rollout.

## Assignment

Students bring a 3–8 second video, select one subject, produce a valid 29DoF reference,
run the smoke profile, and submit:

- source video and selection frame/click point (or rank)
- PromptHMR overlay
- GMR/MjLab validation report
- training command and W&B/TensorBoard plot
- a short analysis of the largest reconstruction or retargeting error
