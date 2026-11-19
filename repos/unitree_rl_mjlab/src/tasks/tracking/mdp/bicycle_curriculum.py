"""Bicycle curriculum: dynamic bike stabilization (Alt A/B) and reward scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_box_plus, quat_apply_inverse

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

import os

STEPS_PER_ITER = 24
# Kinematic→physics schedule (Curr B):
#   0 → FULL_ASSIST: full kinematic lock (playback)
#   FULL_ASSIST → DECAY: linear unlock (physics ramps in)
#   after DECAY: full bike physics (train to max_iterations)
# Default: linear unlock over 0–100k, then full physics (e.g. to 150k).
# Override the duration via BICYCLE_CURRICULUM_DECAY_UNTIL_ITER.
# For resumed runs, BICYCLE_CURRICULUM_START_SCALE can continue from the
# assistance level represented by the checkpoint. START_ITER anchors the new
# decay window to the iteration restored from that checkpoint.
FULL_ASSIST_UNTIL_ITER = 0
DECAY_UNTIL_ITER = int(os.environ.get("BICYCLE_CURRICULUM_DECAY_UNTIL_ITER", "100000"))
START_SCALE = float(os.environ.get("BICYCLE_CURRICULUM_START_SCALE", "1.0"))
START_ITER = int(os.environ.get("BICYCLE_CURRICULUM_START_ITER", "0"))
if not 0.0 <= START_SCALE <= 1.0:
  raise ValueError(
    "BICYCLE_CURRICULUM_START_SCALE must be between 0 and 1, "
    f"got {START_SCALE}"
  )


@dataclass
class BicycleCurriculumState:
  """Mutable curriculum knobs shared by events, rewards, and logging."""

  mode: Literal["A", "B", "C", "bike052", "PD"] = "A"
  start_scale: float = START_SCALE
  stab_scale: float = 1.0
  root_lock_scale: float = 1.0
  steer_lock_scale: float = 1.0
  pelvis_assist_scale: float = 1.0
  contact_weight: float = 3.0
  contact_frc_thres: float = 10.0
  contact_gain: float = 5.0
  # bike052 warmup: linear 1→0 over this many PPO iterations.
  decay_until_iter: float = float(DECAY_UNTIL_ITER)
  pelvis_push_target_speed: float = 1.0
  pelvis_push_start_frame: int = 550
  pelvis_push_end_frame: int = 750
  env_assist_base: torch.Tensor | None = None
  env_assist_scale: torch.Tensor | None = None


def _curriculum_state(env: ManagerBasedRlEnv) -> BicycleCurriculumState:
  state = getattr(env, "_bicycle_curriculum", None)
  if state is None:
    env._bicycle_curriculum = BicycleCurriculumState()
    state = env._bicycle_curriculum
  return state


def init_bicycle_curriculum_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mode: Literal["A", "B", "C", "bike052", "PD"] = "A",
  decay_until_iter: float | None = None,
  pelvis_push_target_speed: float = 1.0,
  pelvis_push_start_frame: int = 550,
  pelvis_push_end_frame: int = 750,
  saddle_contact_local: tuple[float, float, float] | None = None,
  pd_assist_levels: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5, 1.0),
  start_scale: float | None = None,
) -> None:
  """Initialize curriculum state (startup event)."""
  del env_ids
  state = BicycleCurriculumState(mode=mode)
  if start_scale is not None:
    if not 0.0 <= start_scale <= 1.0:
      raise ValueError("start_scale must be in [0, 1]")
    state.start_scale = float(start_scale)
  if decay_until_iter is not None:
    state.decay_until_iter = float(decay_until_iter)
  state.pelvis_push_target_speed = float(pelvis_push_target_speed)
  state.pelvis_push_start_frame = int(pelvis_push_start_frame)
  state.pelvis_push_end_frame = int(pelvis_push_end_frame)
  if mode == "PD":
    if not pd_assist_levels or any(x < 0.0 or x > 1.0 for x in pd_assist_levels):
      raise ValueError("pd_assist_levels must contain values in [0, 1]")
    levels = torch.tensor(pd_assist_levels, device=env.device, dtype=torch.float32)
    repeats = (env.num_envs + levels.numel() - 1) // levels.numel()
    base = levels.repeat(repeats)[: env.num_envs]
    # Decorrelate assistance cohorts spatially while keeping the distribution fixed.
    base = base[torch.randperm(env.num_envs, device=env.device)]
    state.env_assist_base = base
    state.env_assist_scale = base.clone()
    zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    env._bicycle_roll_pd_torque_abs = zeros.clone()
    env._bicycle_steer_pd_torque_abs = zeros.clone()
    env._bicycle_steer_pd_error_abs = zeros.clone()
  env._bicycle_curriculum = state
  if saddle_contact_local is not None:
    env._bicycle_saddle_contact_local = saddle_contact_local


def _training_iter(env: ManagerBasedRlEnv) -> float:
  return env.common_step_counter / STEPS_PER_ITER


def _lerp_stage(it: float, start: float, end: float, v0: float, v1: float) -> float:
  if it <= start:
    return v0
  if it >= end:
    return v1
  t = (it - start) / (end - start)
  return v0 + t * (v1 - v0)


def _update_scales(env: ManagerBasedRlEnv) -> BicycleCurriculumState:
  state = _curriculum_state(env)
  it = _training_iter(env)
  decay_end = (
    state.decay_until_iter
    if state.mode in ("B", "bike052", "PD")
    else float(START_ITER + DECAY_UNTIL_ITER)
  )
  decay_start = float(START_ITER + FULL_ASSIST_UNTIL_ITER)
  decay_scale = _lerp_stage(it, decay_start, decay_end, 1.0, 0.0)
  state.stab_scale = state.start_scale * decay_scale
  if state.mode == "PD":
    # PD assistance applies only a bounded external bicycle roll torque.  It
    # never overwrites bicycle state, locks steering, or pushes the pelvis.
    state.root_lock_scale = 0.0
    state.steer_lock_scale = 0.0
    state.pelvis_assist_scale = 0.0
  else:
    state.root_lock_scale = state.stab_scale
    state.steer_lock_scale = state.stab_scale
    state.pelvis_assist_scale = state.stab_scale
  state.contact_weight = 1.0 + 2.0 * state.stab_scale
  state.contact_frc_thres = 20.0 - 10.0 * state.stab_scale
  state.contact_gain = 5.0
  if state.mode == "PD" and state.env_assist_base is not None:
    state.env_assist_scale = state.env_assist_base * state.stab_scale
  return state


def update_bicycle_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str = "bicycle_eef_contact",
) -> dict[str, torch.Tensor]:
  """Log Curr-B lock scales from ``common_step_counter``.

  EE contact reward weight / frc_thres are fixed (no longer curriculum-driven).
  """
  del env_ids, reward_name
  state = _update_scales(env)
  metrics = {
    "bicycle_stab_scale": torch.tensor(state.stab_scale),
    "bicycle_root_lock": torch.tensor(state.root_lock_scale),
    "bicycle_steer_lock": torch.tensor(state.steer_lock_scale),
    "bicycle_pelvis_assist": torch.tensor(state.pelvis_assist_scale),
  }
  if state.env_assist_scale is not None:
    metrics.update(
      {
        "bicycle_pd_assist_mean": state.env_assist_scale.mean(),
        "bicycle_pd_assist_max": state.env_assist_scale.max(),
        "bicycle_pd_zero_fraction": (state.env_assist_scale <= 1.0e-6).float().mean(),
      }
    )
  for name in (
    "bicycle_roll_pd_torque_abs",
    "bicycle_steer_pd_torque_abs",
    "bicycle_steer_pd_error_abs",
  ):
    value = getattr(env, f"_{name}", None)
    if value is not None:
      metrics[name] = value.mean()
  return metrics


def _require_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.bicycle is None:
    raise RuntimeError(f"{command_name} has no bicycle entity configured")
  return command


def _ref_bike_state(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  bike_qpos = command.motion.bike_qpos[command.time_steps]
  env_origins = command._env.scene.env_origins
  root_pose = bike_qpos[:, :7].clone()
  root_pose[:, :3] += env_origins
  root_velocity, joint_vel = command.reference_bicycle_velocities(command.time_steps)
  joint_pos = bike_qpos[:, 7:]
  return root_pose, root_velocity, joint_pos, joint_vel


def _apply_pelvis_forward_assist(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  scale: float,
  robot_body_name: str = "pelvis",
  assist_gain: float = 80.0,
) -> None:
  if scale <= 1e-4:
    return
  robot = env.scene["robot"]
  body_ids, _ = robot.find_bodies(robot_body_name)
  body_id = body_ids[0]
  pelvis_idx = command.cfg.body_names.index(robot_body_name)
  ref_vel = command.body_lin_vel_w[:, pelvis_idx]
  sim_vel = robot.data.body_link_vel_w[:, body_id, :3]
  vel_err = ref_vel - sim_vel
  force = assist_gain * scale * vel_err
  zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
  robot.write_external_wrench_to_sim(
    force.unsqueeze(1),
    zeros,
    body_ids=[body_id],
  )


def apply_bicycle_curriculum_step_a(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  kp_pos: float = 400.0,
  kd_pos: float = 40.0,
  kp_ori: float = 200.0,
  kd_ori: float = 20.0,
) -> None:
  """Alt A: virtual PD wrench on bicycle frame toward reference."""
  del env_ids
  state = _update_scales(env)
  command = _require_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  frame_ids, _ = bicycle.find_bodies("frame")
  frame_id = frame_ids[0]
  ref_pos = command.ref_bike_pos("frame")
  sim_pos = command.sim_bike_pos("frame")
  ref_i = command._ref_bike_index("frame")
  sim_i = command._sim_bike_index("frame")
  ref_quat = command.ref_bike_body_quat_w[:, ref_i]
  sim_quat = command.sim_bike_body_quat_w[:, sim_i]

  pos_err = ref_pos - sim_pos
  ori_err = quat_box_minus(ref_quat, sim_quat)
  ref_root, ref_root_vel, ref_joint_pos, _ref_joint_vel = _ref_bike_state(command)
  sim_root_vel = bicycle.data.root_link_vel_w
  vel_err = ref_root_vel[:, :3] - sim_root_vel[:, :3]
  ang_vel_err = ref_root_vel[:, 3:] - sim_root_vel[:, 3:]

  scale = state.stab_scale
  force = scale * (kp_pos * pos_err + kd_pos * vel_err)
  torque = scale * (kp_ori * ori_err + kd_ori * ang_vel_err)
  bicycle.write_external_wrench_to_sim(
    force.unsqueeze(1),
    torque.unsqueeze(1),
    body_ids=[frame_id],
  )

  steer_ids, _ = bicycle.find_joints("steer")
  steer_id = steer_ids[0]
  sim_joint_pos = bicycle.data.joint_pos.clone()
  sim_joint_vel = bicycle.data.joint_vel.clone()
  alpha = state.steer_lock_scale
  if alpha > 1e-4:
    sim_joint_pos[:, steer_id] = (
      (1.0 - alpha) * sim_joint_pos[:, steer_id]
      + alpha * ref_joint_pos[:, steer_id]
    )
    sim_joint_vel[:, steer_id] = (
      (1.0 - alpha) * sim_joint_vel[:, steer_id]
      + alpha * _ref_joint_vel[:, steer_id]
    )
    bicycle.write_joint_state_to_sim(
      sim_joint_pos[:, steer_id : steer_id + 1],
      sim_joint_vel[:, steer_id : steer_id + 1],
      joint_ids=[steer_id],
    )

  _apply_pelvis_forward_assist(env, command, state.pelvis_assist_scale)


def apply_bicycle_curriculum_step_b(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
) -> None:
  """Alt B: blend full bicycle state toward kinematic reference.

  ``root_lock_scale == 1`` → full playback (all joints + root overwritten).
  ``root_lock_scale == 0`` → pure dynamic bike physics.
  Intermediate values linearly blend sim ↔ reference each step.
  """
  del env_ids
  state = _update_scales(env)
  command = _require_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  ref_root_pose, ref_root_vel, ref_joint_pos, ref_joint_vel = _ref_bike_state(command)
  alpha = state.root_lock_scale
  if alpha > 1e-4:
    if alpha >= 1.0 - 1e-6:
      # Exact kinematic rewrite (same as BicycleReferenceAction).
      command.write_bicycle_reference()
    else:
      sim_root_pose = bicycle.data.root_link_pose_w.clone()
      sim_root_vel = bicycle.data.root_link_vel_w.clone()
      new_pos = (1.0 - alpha) * sim_root_pose[:, :3] + alpha * ref_root_pose[:, :3]
      delta = quat_box_minus(ref_root_pose[:, 3:7], sim_root_pose[:, 3:7])
      new_quat = quat_box_plus(sim_root_pose[:, 3:7], alpha * delta)
      new_root_pose = torch.cat([new_pos, new_quat], dim=-1)
      new_root_vel = (1.0 - alpha) * sim_root_vel + alpha * ref_root_vel
      bicycle.write_root_link_pose_to_sim(new_root_pose)
      bicycle.write_root_link_velocity_to_sim(new_root_vel)

      # Blend every bike joint (steer / crank / wheels / pedals), not just steer.
      sim_joint_pos = bicycle.data.joint_pos
      sim_joint_vel = bicycle.data.joint_vel
      n_j = min(sim_joint_pos.shape[-1], ref_joint_pos.shape[-1], ref_joint_vel.shape[-1])
      new_jp = (1.0 - alpha) * sim_joint_pos[:, :n_j] + alpha * ref_joint_pos[:, :n_j]
      new_jv = (1.0 - alpha) * sim_joint_vel[:, :n_j] + alpha * ref_joint_vel[:, :n_j]
      bicycle.write_joint_state_to_sim(new_jp, new_jv)

  _apply_pelvis_forward_assist(env, command, state.pelvis_assist_scale)


def apply_bicycle_curriculum_step_c(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  kp_roll: float = 300.0,
  kd_roll: float = 30.0,
  push_force_n: float = 80.0,
  push_start_frame: int = 230,
  push_end_frame: int = 280,
) -> None:
  """Alt C: fully dynamic bike with decaying anti-fall + timed forward push.

  No kinematic lock / reference playback. Wheels and pedals are free (plus
  freewheel clutch). Early training applies:
    1) roll/lean stabilization (keep bike from falling sideways)
    2) forward push on the frame during motion frames
       ``[push_start_frame, push_end_frame]`` (t060 left-foot stamp window)

  Both assists scale with ``stab_scale`` (1 → 0 over the curriculum schedule).
  """
  del env_ids
  state = _update_scales(env)
  scale = state.stab_scale
  command = _require_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  if scale <= 1e-4:
    frame_ids, _ = bicycle.find_bodies("frame")
    zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
    bicycle.write_external_wrench_to_sim(
      zeros, zeros, body_ids=[frame_ids[0]]
    )
    return

  frame_ids, _ = bicycle.find_bodies("frame")
  frame_id = frame_ids[0]
  quat = bicycle.data.body_link_quat_w[:, frame_id]
  ang_vel_w = bicycle.data.root_link_ang_vel_w

  # Body-frame gravity: upright ⇒ g_b ≈ (0, 0, -1). Lateral lean ≈ g_b_y.
  gravity_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
  gravity_w[:, 2] = -1.0
  g_b = quat_apply_inverse(quat, gravity_w)
  lean = g_b[:, 1]

  x_axis = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
  x_axis[:, 0] = 1.0
  forward_w = quat_apply(quat, x_axis)
  roll_rate = (ang_vel_w * forward_w).sum(dim=-1)
  tau_roll = (-kp_roll * lean - kd_roll * roll_rate) * scale
  torque = tau_roll.unsqueeze(-1) * forward_w

  # Timed forward push (same window as left-foot ground stamp).
  frame_mask = (
    (command.time_steps >= push_start_frame) & (command.time_steps <= push_end_frame)
  ).float()
  force = (push_force_n * scale * frame_mask).unsqueeze(-1) * forward_w

  bicycle.write_external_wrench_to_sim(
    force.unsqueeze(1),
    torque.unsqueeze(1),
    body_ids=[frame_id],
  )


def bicycle_eef_contact_exp_curriculum(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_sigma: float = 0.3,
  pos_tolerance: float = 0.0,
  frc_sigma: float = 40.0,
  frc_thres: float = 20.0,
  gain: float = 5.0,
  pedal_z_margin: float = 0.0,
  pedal_z_soft: float = 0.02,
  handle_min_inner_plates: int = 1,
  pair_weights: dict[str, float] | None = None,
) -> torch.Tensor:
  """Deprecated curriculum contact wrapper; prefer ``bicycle_eef_contact_exp``."""
  del frc_thres  # overridden by curriculum state
  from .bicycle_rewards import hdmi_eef_contact_reward

  state = _curriculum_state(env)
  command = _require_command(env, command_name)
  return hdmi_eef_contact_reward(
    env,
    command,
    pos_sigma=pos_sigma,
    pos_tolerance=pos_tolerance,
    frc_sigma=frc_sigma,
    frc_thres=state.contact_frc_thres,
    gain=state.contact_gain,
    pedal_z_margin=pedal_z_margin,
    pedal_z_soft=pedal_z_soft,
    handle_min_inner_plates=handle_min_inner_plates,
    pair_weights=pair_weights,
  )


def _apply_pelvis_world_x_speed_push(
  env: ManagerBasedRlEnv,
  *,
  scale: float,
  target_speed: float,
  start_frame: int,
  end_frame: int,
  command: MotionCommand,
  assist_gain: float = 80.0,
  robot_body_name: str = "pelvis",
) -> None:
  """PD force on pelvis toward world +X speed ``target_speed * scale`` in a frame window.

  Feels like a shove from behind: only when motion time is in
  ``[start_frame, end_frame]`` and ``scale > 0``.
  """
  if scale <= 1e-4:
    return
  robot = env.scene["robot"]
  body_ids, _ = robot.find_bodies(robot_body_name)
  body_id = body_ids[0]
  sim_vel = robot.data.body_link_vel_w[:, body_id, :3]
  frame_mask = (
    (command.time_steps >= start_frame) & (command.time_steps <= end_frame)
  ).float()
  target_vx = target_speed * scale
  vel_err_x = target_vx - sim_vel[:, 0]
  force = torch.zeros(env.num_envs, 3, device=env.device, dtype=sim_vel.dtype)
  force[:, 0] = assist_gain * vel_err_x * frame_mask
  zeros = torch.zeros(env.num_envs, 1, 3, device=env.device, dtype=sim_vel.dtype)
  robot.write_external_wrench_to_sim(
    force.unsqueeze(1),
    zeros,
    body_ids=[body_id],
  )


def apply_bicycle_curriculum_step_bike052(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  kp_roll: float = 300.0,
  kd_roll: float = 30.0,
  pelvis_assist_gain: float = 80.0,
) -> None:
  """bike_052 warmup: dynamic pedals/wheels + decaying roll hold + pelvis +X push.

  - Bicycle is fully dynamic (freewheel clutch elsewhere); no kinematic rewrite.
  - ``stab_scale`` decays 1→0 over ``decay_until_iter`` (default 30k iters).
  - Roll/lean PD keeps the bike upright early, then releases to physics.
  - During motion frames ~11–15 s, push pelvis toward world +X at
    ``pelvis_push_target_speed * stab_scale`` m/s (1→0 m/s over the warmup).
  """
  del env_ids
  state = _update_scales(env)
  scale = state.stab_scale
  command = _require_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  frame_ids, _ = bicycle.find_bodies("frame")
  frame_id = frame_ids[0]
  zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)

  if scale > 1e-4:
    quat = bicycle.data.body_link_quat_w[:, frame_id]
    ang_vel_w = bicycle.data.root_link_ang_vel_w
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
    gravity_w[:, 2] = -1.0
    g_b = quat_apply_inverse(quat, gravity_w)
    lean = g_b[:, 1]
    x_axis = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
    x_axis[:, 0] = 1.0
    forward_w = quat_apply(quat, x_axis)
    roll_rate = (ang_vel_w * forward_w).sum(dim=-1)
    tau_roll = (-kp_roll * lean - kd_roll * roll_rate) * scale
    torque = tau_roll.unsqueeze(-1) * forward_w
    bicycle.write_external_wrench_to_sim(
      zeros, torque.unsqueeze(1), body_ids=[frame_id]
    )
  else:
    bicycle.write_external_wrench_to_sim(zeros, zeros, body_ids=[frame_id])

  _apply_pelvis_world_x_speed_push(
    env,
    scale=scale,
    target_speed=state.pelvis_push_target_speed,
    start_frame=state.pelvis_push_start_frame,
    end_frame=state.pelvis_push_end_frame,
    command=command,
    assist_gain=pelvis_assist_gain,
  )


def apply_bicycle_physical_pd_assistance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "motion",
  kp_roll: float = 120.0,
  kd_roll: float = 20.0,
  max_roll_torque: float = 80.0,
  steer_pd_enabled: bool = False,
  kp_steer: float = 40.0,
  kd_steer: float = 5.0,
  max_steer_torque: float = 15.0,
) -> None:
  """Bounded physical roll PD with optional reference-steering PD.

  Unlike Curriculum B, this never writes bicycle pose, velocity, or joint state.
  Roll assistance applies a world-frame torque about the bicycle's forward
  axis.  Optional steering assistance applies an equal-and-opposite physical
  torque pair to fork and frame about the steer axis, so it creates joint torque
  without injecting net angular momentum.  No pelvis force is applied.
  """
  del env_ids
  state = _update_scales(env)
  command = _require_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  frame_ids, _ = bicycle.find_bodies("frame")
  frame_id = frame_ids[0]
  quat = bicycle.data.body_link_quat_w[:, frame_id]
  ang_vel_w = bicycle.data.root_link_ang_vel_w
  lin_vel_w = bicycle.data.root_link_lin_vel_w

  # Store finite-difference dynamics once per environment step for the optional
  # HAIC-style acceleration observation ablation.
  current_vel = torch.cat((lin_vel_w, ang_vel_w), dim=-1)
  prev_vel = getattr(env, "_bicycle_prev_root_velocity_w", None)
  if prev_vel is None or prev_vel.shape != current_vel.shape:
    prev_vel = current_vel.detach().clone()
  acceleration = (current_vel - prev_vel) / max(float(env.step_dt), 1.0e-6)
  just_reset = env.episode_length_buf <= 1
  acceleration = torch.where(
    just_reset.unsqueeze(-1), torch.zeros_like(acceleration), acceleration
  )
  env._bicycle_root_acceleration_w = acceleration.detach()
  env._bicycle_prev_root_velocity_w = current_vel.detach().clone()

  gravity_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
  gravity_w[:, 2] = -1.0
  lean = quat_apply_inverse(quat, gravity_w)[:, 1]

  ref_idx = command._ref_bike_index("frame")
  ref_quat = command.ref_bike_body_quat_w[:, ref_idx]
  ref_lean = quat_apply_inverse(ref_quat, gravity_w)[:, 1]

  x_axis = torch.zeros(env.num_envs, 3, device=env.device, dtype=quat.dtype)
  x_axis[:, 0] = 1.0
  forward_w = quat_apply(quat, x_axis)
  roll_rate = (ang_vel_w * forward_w).sum(dim=-1)

  if command.motion.bike_body_ang_vel_w is not None:
    ref_ang_vel_w = command.motion.bike_body_ang_vel_w[command.time_steps, ref_idx]
    ref_forward_w = quat_apply(ref_quat, x_axis)
    ref_roll_rate = (ref_ang_vel_w * ref_forward_w).sum(dim=-1)
  else:
    ref_roll_rate = torch.zeros_like(roll_rate)

  scale = state.env_assist_scale
  if scale is None:
    scale = torch.full_like(roll_rate, float(state.stab_scale))
  tau_roll = -kp_roll * (lean - ref_lean) - kd_roll * (roll_rate - ref_roll_rate)
  tau_roll = (tau_roll * scale).clamp(-max_roll_torque, max_roll_torque)
  roll_torque_w = tau_roll.unsqueeze(-1) * forward_w

  fork_ids, _ = bicycle.find_bodies("fork")
  fork_id = fork_ids[0]
  steer_ids, _ = bicycle.find_joints("steer")
  steer_id = steer_ids[0]
  tau_steer = torch.zeros_like(tau_roll)
  steer_error = torch.zeros_like(tau_roll)
  if steer_pd_enabled:
    steer = bicycle.data.joint_pos[:, steer_id]
    steer_rate = bicycle.data.joint_vel[:, steer_id]
    ref_joint_pos = command.motion.bike_qpos[command.time_steps, 7:]
    _, ref_joint_vel = command.reference_bicycle_velocities(command.time_steps)
    ref_steer = ref_joint_pos[:, steer_id]
    ref_steer_rate = ref_joint_vel[:, steer_id]
    steer_error = steer - ref_steer
    tau_steer = -kp_steer * steer_error - kd_steer * (
      steer_rate - ref_steer_rate
    )
    tau_steer = (tau_steer * scale).clamp(
      -max_steer_torque, max_steer_torque
    )

  # The steer hinge axis is expressed in the frame body coordinates in XML.
  # Applying +tau to the fork and -tau to the frame is an internal joint torque.
  steer_axis_b = torch.zeros_like(forward_w)
  steer_axis_b[:, 0] = -0.34202
  steer_axis_b[:, 2] = 0.93969
  steer_axis_w = quat_apply(quat, steer_axis_b)
  steer_torque_w = tau_steer.unsqueeze(-1) * steer_axis_w
  frame_torque_w = roll_torque_w - steer_torque_w
  torques = torch.stack((frame_torque_w, steer_torque_w), dim=1)
  zeros = torch.zeros(env.num_envs, 2, 3, device=env.device, dtype=quat.dtype)
  bicycle.write_external_wrench_to_sim(
    zeros, torques, body_ids=[frame_id, fork_id]
  )
  env._bicycle_roll_pd_torque_abs = tau_roll.abs().detach()
  env._bicycle_steer_pd_torque_abs = tau_steer.abs().detach()
  env._bicycle_steer_pd_error_abs = steer_error.abs().detach()
