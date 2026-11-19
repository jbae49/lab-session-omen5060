"""Pelvis→saddle hard lock with linear unlock curriculum (for seated pedaling POC)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_box_minus,
  quat_box_plus,
  quat_mul,
)

from .bicycle_data import SADDLE60_LOCAL_POS, SADDLE_PARENT_BODY
from .bicycle_curriculum import STEPS_PER_ITER, _lerp_stage
from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DECAY_UNTIL_ITER = int(os.environ.get("PELVIS_SADDLE_LOCK_DECAY_UNTIL_ITER", "30000"))
START_SCALE = float(os.environ.get("PELVIS_SADDLE_LOCK_START_SCALE", "1.0"))


def init_pelvis_saddle_lock(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  decay_until_iter: float | None = None,
  saddle_local_pos: tuple[float, float, float] = SADDLE60_LOCAL_POS,
) -> None:
  """Cache saddle←pelvis offset from motion frame 0 (cut already locked)."""
  del env_ids
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.motion.bike_body_pos_w is None:
    raise RuntimeError("pelvis-saddle lock requires bike bodies in motion npz")

  # Use first motion frame as anchor (cut export locks at original frame 375).
  t0 = torch.zeros(1, dtype=torch.long, device=env.device)
  frame_i = command._ref_bike_index(SADDLE_PARENT_BODY)
  saddle_pos = command.motion.bike_body_pos_w[t0, frame_i][0].clone()
  saddle_quat = command.motion.bike_body_quat_w[t0, frame_i][0].clone()
  # Apply saddle local offset in frame body.
  local = torch.tensor(saddle_local_pos, device=env.device, dtype=saddle_pos.dtype)
  saddle_pos = saddle_pos + quat_apply(saddle_quat.unsqueeze(0), local.unsqueeze(0))[0]

  pelvis_i = command.cfg.body_names.index("pelvis")
  pelvis_pos = command.motion.body_pos_w[t0, pelvis_i][0].clone()
  pelvis_quat = command.motion.body_quat_w[t0, pelvis_i][0].clone()

  # pos/rot in saddle frame
  pos_sp = quat_apply_inverse(
    saddle_quat.unsqueeze(0), (pelvis_pos - saddle_pos).unsqueeze(0)
  )[0]
  # q_sp = inv(q_s) * q_p
  q_sp = quat_mul(
    _quat_conjugate(saddle_quat.unsqueeze(0)), pelvis_quat.unsqueeze(0)
  )[0]

  env._pelvis_saddle_lock = {
    "pos_sp": pos_sp,
    "quat_sp": q_sp,
    "saddle_local": local,
    "decay_until_iter": float(
      DECAY_UNTIL_ITER if decay_until_iter is None else decay_until_iter
    ),
    "start_scale": float(START_SCALE),
    "lock_scale": float(START_SCALE),
  }


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  out = q.clone()
  out[..., 1:] = -out[..., 1:]
  return out


def _lock_scale(env: ManagerBasedRlEnv) -> float:
  state = getattr(env, "_pelvis_saddle_lock", None)
  if state is None:
    return 0.0
  it = env.common_step_counter / STEPS_PER_ITER
  decay_end = float(state["decay_until_iter"])
  scale = state["start_scale"] * _lerp_stage(it, 0.0, decay_end, 1.0, 0.0)
  state["lock_scale"] = scale
  return scale


def apply_pelvis_saddle_lock_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
) -> None:
  """Blend robot root toward saddle⊗offset; α=1 full lock → α=0 free."""
  del env_ids
  state = getattr(env, "_pelvis_saddle_lock", None)
  if state is None:
    return
  alpha = _lock_scale(env)
  if alpha <= 1e-4:
    return

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  robot = command.robot
  bicycle = command.bicycle
  assert bicycle is not None

  frame_ids, _ = bicycle.find_bodies(SADDLE_PARENT_BODY)
  frame_id = frame_ids[0]
  frame_pos = bicycle.data.body_link_pos_w[:, frame_id]
  frame_quat = bicycle.data.body_link_quat_w[:, frame_id]
  local = state["saddle_local"].expand(env.num_envs, 3)
  saddle_pos = frame_pos + quat_apply(frame_quat, local)
  saddle_quat = frame_quat

  pos_sp = state["pos_sp"].expand(env.num_envs, 3)
  quat_sp = state["quat_sp"].expand(env.num_envs, 4)
  target_pos = saddle_pos + quat_apply(saddle_quat, pos_sp)
  target_quat = quat_mul(saddle_quat, quat_sp)

  # Match bicycle kinematic velocity (root) so lock is not jerking.
  bike_root_vel = bicycle.data.root_link_vel_w
  target_root_pose = torch.cat([target_pos, target_quat], dim=-1)
  target_root_vel = bike_root_vel

  if alpha >= 1.0 - 1e-6:
    robot.write_root_link_pose_to_sim(target_root_pose)
    robot.write_root_link_velocity_to_sim(target_root_vel)
    return

  sim_pose = robot.data.root_link_pose_w.clone()
  sim_vel = robot.data.root_link_vel_w.clone()
  new_pos = (1.0 - alpha) * sim_pose[:, :3] + alpha * target_pos
  delta = quat_box_minus(target_quat, sim_pose[:, 3:7])
  new_quat = quat_box_plus(sim_pose[:, 3:7], alpha * delta)
  new_vel = (1.0 - alpha) * sim_vel + alpha * target_root_vel
  robot.write_root_link_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1))
  robot.write_root_link_velocity_to_sim(new_vel)


def update_pelvis_saddle_lock_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
  del env_ids
  scale = _lock_scale(env)
  return {"pelvis_saddle_lock_scale": torch.tensor(scale, device=env.device)}
