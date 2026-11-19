#!/usr/bin/env python3
"""Record a trained policy rollout into a new mjlab bicycle motion npz.

Keeps bicycle arrays + contact flags from the source motion (does not overwrite
the source file). Replaces robot joint/body trajectories with the simulated
policy rollout — useful when kinematic-bike training produces a better robot
path than the original reference.

Example:
  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl python scripts/record_policy_motion.py \\
    --task Unitree-G1-Bicycle-Tracking-No-State-Estimation \\
    --motion-file src/assets/motions/g1/..._reg3.npz \\
    --checkpoint-file logs/rsl_rl/g1_tracking/.../model_38000.pt \\
    --output src/assets/motions/g1/..._policy_rollout.npz
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import src.tasks  # noqa: F401 — register tasks
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.tracking.mdp.commands import MotionCommand


ROBOT_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-Bicycle-Tracking-No-State-Estimation",
  )
  p.add_argument("--motion-file", type=Path, required=True)
  p.add_argument("--checkpoint-file", type=Path, required=True)
  p.add_argument(
    "--output",
    type=Path,
    default=None,
    help="New npz path (never overwrites source). Default: <motion>_policy_rollout_<ts>.npz",
  )
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument(
    "--num-frames",
    type=int,
    default=None,
    help="Frames to record (default: full source motion length).",
  )
  p.add_argument(
    "--force",
    action="store_true",
    help="Allow overwriting an existing --output file (still never the source).",
  )
  return p.parse_args()


def _default_output(motion_file: Path) -> Path:
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  return motion_file.with_name(f"{motion_file.stem}_policy_rollout_{ts}.npz")


@torch.no_grad()
def record_rollout(
  task: str,
  motion_file: Path,
  checkpoint_file: Path,
  device: str,
  num_frames: int | None,
) -> dict[str, np.ndarray]:
  configure_torch_backends()
  src = dict(np.load(motion_file, allow_pickle=True))
  T_src = int(src["joint_pos"].shape[0])
  T = int(num_frames) if num_frames is not None else T_src
  if T <= 0:
    raise ValueError(f"num_frames must be > 0, got {T}")
  if T > T_src:
    print(f"[WARN] num_frames={T} > source length {T_src}; clamping")
    T = T_src

  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = 1
  env_cfg.terminations = {}
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)

  motion_cmd = env_cfg.commands["motion"]
  motion_cmd.motion_file = str(motion_file.resolve())
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.sampling_mode = "start"
  motion_cmd.debug_vis = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_file),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  raw = env.unwrapped
  robot = raw.scene["robot"]
  cmd = raw.command_manager.get_term("motion")
  assert isinstance(cmd, MotionCommand)

  obs, _ = env.reset()
  # Ensure clip starts at frame 0 with no residual pose noise.
  cmd.time_steps[:] = 0
  cmd._resample_command(torch.arange(raw.num_envs, device=raw.device))
  raw.sim.forward()
  obs = env.get_observations()

  logs: dict[str, list[np.ndarray]] = {k: [] for k in ROBOT_KEYS}
  origin = raw.scene.env_origins[0].detach()

  for t in range(T):
    # Record state *before* step so frame t aligns with motion time_steps==t.
    logs["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy())
    logs["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy())
    logs["body_pos_w"].append(
      (robot.data.body_link_pos_w[0] - origin[None, :]).detach().cpu().numpy()
    )
    logs["body_quat_w"].append(robot.data.body_link_quat_w[0].detach().cpu().numpy())
    logs["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0].detach().cpu().numpy()
    )
    logs["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0].detach().cpu().numpy()
    )

    if t + 1 >= T:
      break
    actions = policy(obs)
    step_out = env.step(actions)
    obs = step_out[0]

    # If motion resampled/wrapped early, stop.
    if int(cmd.time_steps[0].item()) == 0 and t > 0:
      print(f"[WARN] motion wrapped at step {t}; truncating record")
      for k in logs:
        logs[k] = logs[k][:t]
      break

  env.close()

  out = dict(src)
  for k in ROBOT_KEYS:
    out[k] = np.stack(logs[k], axis=0).astype(np.float32)

  # Trim shared arrays to recorded length; keep bike + contact from source.
  T_out = out["joint_pos"].shape[0]
  for k, v in list(out.items()):
    if k in ("fps", "bike_body_names", "bike_contact_pair_keys", "crank_sign"):
      continue
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T_src:
      out[k] = v[:T_out]

  robot_root = np.concatenate(
    [out["body_pos_w"][:, 0], out["body_quat_w"][:, 0], out["joint_pos"]],
    axis=1,
  ).astype(np.float32)
  if "bike_qpos" in out:
    out["combined_qpos"] = np.concatenate(
      [out["bike_qpos"].astype(np.float32), robot_root], axis=1
    ).astype(np.float32)

  out["fps"] = np.asarray(src["fps"]).reshape(-1).astype(np.float64)
  out["source_motion_file"] = np.asarray(str(motion_file.resolve()))
  out["source_checkpoint"] = np.asarray(str(checkpoint_file.resolve()))
  out["recorded_task"] = np.asarray(task)
  return out


def main() -> None:
  args = _parse_args()
  motion_file = args.motion_file.resolve()
  checkpoint_file = args.checkpoint_file.resolve()
  if not motion_file.exists():
    raise FileNotFoundError(motion_file)
  if not checkpoint_file.exists():
    raise FileNotFoundError(checkpoint_file)

  output = (args.output or _default_output(motion_file)).resolve()
  if output == motion_file:
    raise ValueError("Refusing to overwrite the source motion file; pick another --output")
  if output.exists() and not args.force:
    raise FileExistsError(f"Output exists (pass --force to overwrite): {output}")

  print(f"[INFO] Recording policy motion")
  print(f"  task:       {args.task}")
  print(f"  motion:     {motion_file}")
  print(f"  checkpoint: {checkpoint_file}")
  print(f"  output:     {output}")

  out = record_rollout(
    task=args.task,
    motion_file=motion_file,
    checkpoint_file=checkpoint_file,
    device=args.device,
    num_frames=args.num_frames,
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **out)
  print(
    f"[OK] Saved {output}\n"
    f"     frames={out['joint_pos'].shape[0]}  "
    f"joint_pos={out['joint_pos'].shape}  "
    f"bike_kept_from_source={('bike_qpos' in out)}"
  )


if __name__ == "__main__":
  main()
