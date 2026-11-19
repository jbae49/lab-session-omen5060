"""Observations for bicycle interaction tracking (incl. privileged terms)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .bicycle_data import (
  BICYCLE_CONTACT_PAIR_KEYS,
  BICYCLE_CONTACT_PAIRS,
  BICYCLE_OBS_BODIES,
  BICYCLE_TRACK_BODY,
  BICYCLE_VEL_OBS_BODIES,
  HANDLE_HAND_L_LOCAL,
  HANDLE_HAND_R_LOCAL,
  HANDLE_LOCAL_POS,
  HANDLE_PARENT_BODY,
  PEDAL_L_CONTACT_LOCAL,
  PEDAL_R_CONTACT_LOCAL,
  SADDLE_LOCAL_POS,
  SADDLE_PARENT_BODY,
)

SaddleLocalPos = tuple[float, float, float]
from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
PEDAL_BODIES: tuple[str, ...] = ("pedal_left", "pedal_right")
# Contact privilege: interaction pairs (exclude foot–ground).
CONTACT_PRIV_KEYS: tuple[str, ...] = (
  "handle_left",
  "handle_right",
  "saddle",
  "pedal_left",
  "pedal_right",
)
# Rigid bodies whose ori6 is exposed (handle L/R share fork; saddle uses frame).
BIKE_COMPONENT_ORI_BODIES: tuple[str, ...] = (
  "fork",
  "frame",
  "pedal_left",
  "pedal_right",
)


def _require_bicycle_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.bicycle is None:
    raise RuntimeError(f"{command_name} has no bicycle entity configured")
  return command


def _pos_b(command: MotionCommand, pos_w: torch.Tensor) -> torch.Tensor:
  """World position(s) → robot-anchor frame. ``pos_w``: (B, 3) or (B, N, 3)."""
  if pos_w.ndim == 2:
    return quat_apply_inverse(
      command.robot_anchor_quat_w, pos_w - command.robot_anchor_pos_w
    )
  B, N, _ = pos_w.shape
  q = command.robot_anchor_quat_w[:, None, :].expand(B, N, 4).reshape(B * N, 4)
  t = command.robot_anchor_pos_w[:, None, :].expand(B, N, 3)
  return quat_apply_inverse(q, (pos_w - t).reshape(B * N, 3)).reshape(B, N, 3)


def _vel_b(command: MotionCommand, vel_w: torch.Tensor) -> torch.Tensor:
  """World linear velocity(s) → robot-anchor frame (rotate only)."""
  if vel_w.ndim == 2:
    return quat_apply_inverse(command.robot_anchor_quat_w, vel_w)
  B, N, _ = vel_w.shape
  q = command.robot_anchor_quat_w[:, None, :].expand(B, N, 4).reshape(B * N, 4)
  return quat_apply_inverse(q, vel_w.reshape(B * N, 3)).reshape(B, N, 3)


def _flat(x: torch.Tensor) -> torch.Tensor:
  return x.reshape(x.shape[0], -1)


def reference_motion_phase(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """One-shot clip progress plus cyclic encoding: ``[phi, sin, cos]``. (B, 3)."""
  command = _require_bicycle_command(env, command_name)
  denom = max(command.motion.time_step_total - 1, 1)
  phase = command.time_steps.float() / float(denom)
  angle = 2.0 * torch.pi * phase
  return torch.stack((phase, torch.sin(angle), torch.cos(angle)), dim=-1)


def bicycle_crank_phase(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual crank phase as a wrap-safe sine/cosine pair. (B, 2)."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  crank_ids, _ = command.bicycle.find_joints("crank_spin")
  angle = command.bicycle.data.joint_pos[:, crank_ids[0]]
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def bicycle_contact_time_to_transition(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_horizon_s: float = 5.0,
) -> torch.Tensor:
  """Normalized time until each interaction contact flag changes. (B, 5)."""
  command = _require_bicycle_command(env, command_name)
  cache = getattr(command, "_contact_transition_frames", None)
  if cache is None:
    flags = command._ref_contact_mask
    T, N = flags.shape
    cache = torch.zeros(T, N, device=flags.device, dtype=torch.float32)
    for pair_idx in range(N):
      next_change = T - 1
      cache[T - 1, pair_idx] = 0.0
      for frame in range(T - 2, -1, -1):
        if flags[frame, pair_idx] != flags[frame + 1, pair_idx]:
          next_change = frame + 1
        cache[frame, pair_idx] = float(next_change - frame)
    command._contact_transition_frames = cache
  idxs = [BICYCLE_CONTACT_PAIR_KEYS.index(k) for k in CONTACT_PRIV_KEYS]
  seconds = cache[command.time_steps][:, idxs] / float(command.motion.fps)
  return (seconds / max(float(max_horizon_s), 1.0e-6)).clamp(0.0, 1.0)


def bicycle_root_velocity_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual bicycle root linear and angular velocity in robot-anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  return torch.cat(
    (
      _vel_b(command, command.bicycle.data.root_link_lin_vel_w),
      _vel_b(command, command.bicycle.data.root_link_ang_vel_w),
    ),
    dim=-1,
  )


def bicycle_root_velocity_ref_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reference bicycle root linear and angular velocity in robot frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  root_vel, _ = command.reference_bicycle_velocities(command.time_steps)
  return torch.cat(
    (_vel_b(command, root_vel[:, :3]), _vel_b(command, root_vel[:, 3:])), dim=-1
  )


def bicycle_deployable_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current bicycle state measurable with an IMU and joint encoders. (B, 8).

  Order:
    lean angle, roll rate, steer angle, steer rate,
    sin(crank angle), cos(crank angle), signed crank rate, rear-wheel rate.

  This deliberately contains no reference/ghost state and no simulator contact
  force.  In hardware, lean/roll-rate come from a bicycle-frame IMU and the
  remaining values from steering, crank, and rear-wheel encoders.
  """
  command = _require_bicycle_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  frame_i = command._sim_bike_index("frame")
  frame_quat = command.sim_bike_body_quat_w[:, frame_i]
  gravity_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=frame_quat.dtype)
  gravity_w[:, 2] = -1.0
  gravity_b = quat_apply_inverse(frame_quat, gravity_w)
  # Upright: gravity_b=[0, 0, -1].  atan2 is well behaved for the expected
  # bicycle pitch range and returns a signed roll/lean angle in radians.
  lean = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])

  x_axis = torch.zeros_like(gravity_w)
  x_axis[:, 0] = 1.0
  forward_w = quat_apply(frame_quat, x_axis)
  roll_rate = (bicycle.data.root_link_ang_vel_w * forward_w).sum(dim=-1)

  steer_ids, _ = bicycle.find_joints("steer")
  crank_ids, _ = bicycle.find_joints("crank_spin")
  rear_ids, _ = bicycle.find_joints("rear_spin")
  jp = bicycle.data.joint_pos
  jv = bicycle.data.joint_vel
  steer = jp[:, steer_ids[0]]
  steer_rate = jv[:, steer_ids[0]]
  crank = jp[:, crank_ids[0]]
  crank_rate = jv[:, crank_ids[0]]
  rear_rate = jv[:, rear_ids[0]]
  return torch.stack(
    (
      lean,
      roll_rate,
      steer,
      steer_rate,
      torch.sin(crank),
      torch.cos(crank),
      crank_rate,
      rear_rate,
    ),
    dim=-1,
  )


def bicycle_root_acceleration_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Finite-difference bicycle linear/angular acceleration in robot frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  acc_w = getattr(env, "_bicycle_root_acceleration_w", None)
  if acc_w is None:
    acc_w = torch.zeros(command.num_envs, 6, device=command.device)
  return torch.cat(
    (_vel_b(command, acc_w[:, :3]), _vel_b(command, acc_w[:, 3:])), dim=-1
  )


def bicycle_pd_assistance_scale(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Per-environment bicycle assistance scale. (B, 1).

  PD mode reports its per-environment roll-PD cohort scale.  Kinematic
  curricula such as Curr-B report the shared root/joint blend alpha.
  """
  del command_name
  state = getattr(env, "_bicycle_curriculum", None)
  scale = getattr(state, "env_assist_scale", None)
  if scale is not None:
    return scale.unsqueeze(-1)
  alpha = float(getattr(state, "root_lock_scale", 0.0))
  return torch.full((env.num_envs, 1), alpha, device=env.device)


def _ref_foot_pos_w(command: MotionCommand) -> torch.Tensor:
  idx = [command.cfg.body_names.index(n) for n in FOOT_BODIES]
  return command.body_pos_w[:, idx]


def _sim_foot_pos_w(command: MotionCommand) -> torch.Tensor:
  idx = [command.cfg.body_names.index(n) for n in FOOT_BODIES]
  return command.robot_body_pos_w[:, idx]


def _ref_foot_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  idx = [command.cfg.body_names.index(n) for n in FOOT_BODIES]
  return command.body_lin_vel_w[:, idx]


def _sim_foot_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  idx = [command.cfg.body_names.index(n) for n in FOOT_BODIES]
  return command.robot_body_lin_vel_w[:, idx]


def _ref_pedal_pos_w(command: MotionCommand) -> torch.Tensor:
  idx = [command._ref_bike_index(n) for n in PEDAL_BODIES]
  return command.ref_bike_body_pos_w[:, idx]


def _sim_pedal_pos_w(command: MotionCommand) -> torch.Tensor:
  idx = [command._sim_bike_index(n) for n in PEDAL_BODIES]
  return command.sim_bike_body_pos_w[:, idx]


def _ref_pedal_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  if command.motion.bike_body_lin_vel_w is None:
    return torch.zeros(
      command.num_envs, len(PEDAL_BODIES), 3, device=command.device
    )
  idx = [command._ref_bike_index(n) for n in PEDAL_BODIES]
  return command.motion.bike_body_lin_vel_w[command.time_steps][:, idx]


def _sim_pedal_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  assert command.bicycle is not None
  idx = [command._sim_bike_index(n) for n in PEDAL_BODIES]
  ids = command._sim_bike_body_ids[idx]
  return command.bicycle.data.body_link_lin_vel_w[:, ids]


def _contact_force_mag(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  if force.ndim == 3:
    return torch.norm(force, dim=-1).amax(dim=-1)
  return torch.norm(force, dim=-1)


# ---------------------------------------------------------------------------
# Legacy (kept for compatibility)
# ---------------------------------------------------------------------------


def bicycle_ref_points_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference pedal/handle/saddle points in the robot anchor frame.

  Uses motion reference (command) — same information style as ``command`` obs.
  Shape: (num_envs, 3 * len(BICYCLE_OBS_BODIES)).
  """
  command = _require_bicycle_command(env, command_name)
  parts: list[torch.Tensor] = []
  for body_name in BICYCLE_OBS_BODIES:
    pos_b, _ = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      command.ref_bike_pos(body_name),
      command.ref_bike_body_quat_w[:, command._ref_bike_index(body_name)],
    )
    parts.append(pos_b)
  return torch.cat(parts, dim=-1)


def bicycle_sim_points_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Simulated bicycle interaction points in the robot anchor frame (critic)."""
  command = _require_bicycle_command(env, command_name)
  parts: list[torch.Tensor] = []
  for body_name in BICYCLE_OBS_BODIES:
    pos_b, _ = subtract_frame_transforms(
      command.robot_anchor_pos_w,
      command.robot_anchor_quat_w,
      command.sim_bike_pos(body_name),
      command.sim_bike_body_quat_w[:, command._sim_bike_index(body_name)],
    )
    parts.append(pos_b)
  return torch.cat(parts, dim=-1)


def _ref_bike_component_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  """Ref lin vel for ``BICYCLE_VEL_OBS_BODIES``. (B, K, 3)."""
  K = len(BICYCLE_VEL_OBS_BODIES)
  if command.motion.bike_body_lin_vel_w is None:
    return torch.zeros(command.num_envs, K, 3, device=command.device)
  idx = [command._ref_bike_index(n) for n in BICYCLE_VEL_OBS_BODIES]
  return command.motion.bike_body_lin_vel_w[command.time_steps][:, idx]


def _sim_bike_component_lin_vel_w(command: MotionCommand) -> torch.Tensor:
  """Sim lin vel for ``BICYCLE_VEL_OBS_BODIES``. (B, K, 3)."""
  assert command.bicycle is not None
  idx = [command._sim_bike_index(n) for n in BICYCLE_VEL_OBS_BODIES]
  ids = command._sim_bike_body_ids[idx]
  return command.bicycle.data.body_link_lin_vel_w[:, ids]


def bicycle_ref_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref bike component linear velocities in robot-anchor frame. (B, 3*K)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_vel_b(command, _ref_bike_component_lin_vel_w(command)))


def bicycle_sim_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim bike component linear velocities in robot-anchor frame. (B, 3*K)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_vel_b(command, _sim_bike_component_lin_vel_w(command)))


def _local_offset_w(
  parent_pos_w: torch.Tensor,
  parent_quat_w: torch.Tensor,
  local_pos: tuple[float, float, float],
) -> torch.Tensor:
  """Parent pose + body-local offset → world position. (B, 3)."""
  offset = torch.tensor(local_pos, dtype=parent_pos_w.dtype, device=parent_pos_w.device)
  offset = offset.unsqueeze(0).expand(parent_pos_w.shape[0], -1)
  return parent_pos_w + quat_apply(parent_quat_w, offset)


def _ref_interaction_point_w(
  command: MotionCommand,
  parent_body: str,
  local_pos: tuple[float, float, float],
) -> torch.Tensor:
  idx = command._ref_bike_index(parent_body)
  return _local_offset_w(
    command.ref_bike_body_pos_w[:, idx],
    command.ref_bike_body_quat_w[:, idx],
    local_pos,
  )


def _sim_interaction_point_w(
  command: MotionCommand,
  parent_body: str,
  local_pos: tuple[float, float, float],
) -> torch.Tensor:
  idx = command._sim_bike_index(parent_body)
  return _local_offset_w(
    command.sim_bike_body_pos_w[:, idx],
    command.sim_bike_body_quat_w[:, idx],
    local_pos,
  )


def bike_handle_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim handlebar midpoint in robot-anchor frame. (B, 3)."""
  command = _require_bicycle_command(env, command_name)
  return _pos_b(
    command,
    _sim_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_LOCAL_POS),
  )


def bike_handle_pos_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref handlebar midpoint in robot-anchor frame. (B, 3)."""
  command = _require_bicycle_command(env, command_name)
  return _pos_b(
    command,
    _ref_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_LOCAL_POS),
  )


def bike_saddle_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Sim saddle center in robot-anchor frame. (B, 3)."""
  command = _require_bicycle_command(env, command_name)
  return _pos_b(
    command,
    _sim_interaction_point_w(command, SADDLE_PARENT_BODY, saddle_local_pos),
  )


def bike_saddle_pos_ref_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Ref saddle center in robot-anchor frame. (B, 3)."""
  command = _require_bicycle_command(env, command_name)
  return _pos_b(
    command,
    _ref_interaction_point_w(command, SADDLE_PARENT_BODY, saddle_local_pos),
  )


def _contact_interaction_points_w(
  command: MotionCommand,
  *,
  sim: bool,
  saddle_local_pos: SaddleLocalPos,
) -> torch.Tensor:
  """World contact targets. (B, 5, 3): handleL, handleR, saddle, pedalL, pedalR."""
  point_fn = _sim_interaction_point_w if sim else _ref_interaction_point_w
  handle_l = point_fn(command, HANDLE_PARENT_BODY, HANDLE_HAND_L_LOCAL)
  handle_r = point_fn(command, HANDLE_PARENT_BODY, HANDLE_HAND_R_LOCAL)
  saddle = point_fn(command, SADDLE_PARENT_BODY, saddle_local_pos)
  pedal_l = point_fn(command, "pedal_left", PEDAL_L_CONTACT_LOCAL)
  pedal_r = point_fn(command, "pedal_right", PEDAL_R_CONTACT_LOCAL)
  return torch.stack((handle_l, handle_r, saddle, pedal_l, pedal_r), dim=1)


def bike_interaction_points_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Sim contact points in robot-anchor frame. (B, 15).

  Order: handleL, handleR, saddle, pedalL, pedalR — pos3 each.
  Handles use ``HANDLE_HAND_*``; pedals use ``PEDAL_*_CONTACT_LOCAL``.
  """
  command = _require_bicycle_command(env, command_name)
  pts = _contact_interaction_points_w(
    command, sim=True, saddle_local_pos=saddle_local_pos
  )
  return _flat(_pos_b(command, pts))


def bike_interaction_points_ref_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Ref contact points in robot-anchor frame. (B, 15). Same order as sim."""
  command = _require_bicycle_command(env, command_name)
  pts = _contact_interaction_points_w(
    command, sim=False, saddle_local_pos=saddle_local_pos
  )
  return _flat(_pos_b(command, pts))


def bike_interaction_points_b_legacy12(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Legacy Hist9-Rich pack: handleL/R + pedalL/R only. (B, 12)."""
  del saddle_local_pos
  command = _require_bicycle_command(env, command_name)
  pts = torch.stack(
    (
      _sim_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_HAND_L_LOCAL),
      _sim_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_HAND_R_LOCAL),
      _sim_interaction_point_w(command, "pedal_left", PEDAL_L_CONTACT_LOCAL),
      _sim_interaction_point_w(command, "pedal_right", PEDAL_R_CONTACT_LOCAL),
    ),
    dim=1,
  )
  return _flat(_pos_b(command, pts))


def bike_interaction_points_ref_b_legacy12(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Legacy Hist9-Rich ref pack: handleL/R + pedalL/R. (B, 12)."""
  del saddle_local_pos
  command = _require_bicycle_command(env, command_name)
  pts = torch.stack(
    (
      _ref_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_HAND_L_LOCAL),
      _ref_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_HAND_R_LOCAL),
      _ref_interaction_point_w(command, "pedal_left", PEDAL_L_CONTACT_LOCAL),
      _ref_interaction_point_w(command, "pedal_right", PEDAL_R_CONTACT_LOCAL),
    ),
    dim=1,
  )
  return _flat(_pos_b(command, pts))


def _ori6_b(
  command: MotionCommand, pos_w: torch.Tensor, quat_w: torch.Tensor
) -> torch.Tensor:
  """Component orientation relative to robot anchor as 6D rotation. (B, 6)."""
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    pos_w,
    quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _bike_component_ori_pack(
  command: MotionCommand, *, sim: bool
) -> torch.Tensor:
  """Ori6 pack for ``BIKE_COMPONENT_ORI_BODIES``. (B, 24)."""
  if sim:
    pos_w = command.sim_bike_body_pos_w
    quat_w = command.sim_bike_body_quat_w
    index_fn = command._sim_bike_index
  else:
    pos_w = command.ref_bike_body_pos_w
    quat_w = command.ref_bike_body_quat_w
    index_fn = command._ref_bike_index
  parts = []
  for name in BIKE_COMPONENT_ORI_BODIES:
    idx = index_fn(name)
    parts.append(_ori6_b(command, pos_w[:, idx], quat_w[:, idx]))
  return torch.cat(parts, dim=-1)


def bike_component_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim bike rigid-body orientations in robot-anchor frame. (B, 24).

  Order: fork, frame, pedal_left, pedal_right — 6D each.
  Handle L/R share fork; saddle uses frame (no duplicate ori).
  """
  command = _require_bicycle_command(env, command_name)
  return _bike_component_ori_pack(command, sim=True)


def bike_component_ori_ref_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Ref bike rigid-body orientations in robot-anchor frame. (B, 24)."""
  command = _require_bicycle_command(env, command_name)
  return _bike_component_ori_pack(command, sim=False)


def bike_component_pose_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Legacy ResMimic pack: 4×(pos3+ori6)=36D (mid-handle + pedal contact).

  Prefer ``bike_interaction_points_b`` + ``bike_component_ori_b`` for new runs.
  Order: pedal_left, pedal_right, handle (fork mid), saddle.
  """
  command = _require_bicycle_command(env, command_name)

  pedal_l_pos = _sim_interaction_point_w(
    command, "pedal_left", PEDAL_L_CONTACT_LOCAL
  )
  pedal_r_pos = _sim_interaction_point_w(
    command, "pedal_right", PEDAL_R_CONTACT_LOCAL
  )
  handle_pos = _sim_interaction_point_w(
    command, HANDLE_PARENT_BODY, HANDLE_LOCAL_POS
  )
  saddle_pos = _sim_interaction_point_w(
    command, SADDLE_PARENT_BODY, saddle_local_pos
  )

  pedal_l_quat = command.sim_bike_body_quat_w[
    :, command._sim_bike_index("pedal_left")
  ]
  pedal_r_quat = command.sim_bike_body_quat_w[
    :, command._sim_bike_index("pedal_right")
  ]
  handle_quat = command.sim_bike_body_quat_w[
    :, command._sim_bike_index(HANDLE_PARENT_BODY)
  ]
  saddle_quat = command.sim_bike_body_quat_w[
    :, command._sim_bike_index(SADDLE_PARENT_BODY)
  ]

  parts = []
  for pos_w, quat_w in (
    (pedal_l_pos, pedal_l_quat),
    (pedal_r_pos, pedal_r_quat),
    (handle_pos, handle_quat),
    (saddle_pos, saddle_quat),
  ):
    parts.append(_pos_b(command, pos_w))
    parts.append(_ori6_b(command, pos_w, quat_w))
  return torch.cat(parts, dim=-1)


def bike_component_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  saddle_local_pos: SaddleLocalPos = SADDLE_LOCAL_POS,
) -> torch.Tensor:
  """Sim bike component centers in robot-anchor frame. (B, 12).

  Order: pedal_left, pedal_right, handle, saddle — pos3 each.
  """
  command = _require_bicycle_command(env, command_name)
  pts = torch.stack(
    (
      _sim_interaction_point_w(command, "pedal_left", PEDAL_L_CONTACT_LOCAL),
      _sim_interaction_point_w(command, "pedal_right", PEDAL_R_CONTACT_LOCAL),
      _sim_interaction_point_w(command, HANDLE_PARENT_BODY, HANDLE_LOCAL_POS),
      _sim_interaction_point_w(command, SADDLE_PARENT_BODY, saddle_local_pos),
    ),
    dim=1,
  )
  return _flat(_pos_b(command, pts))


def bicycle_ref_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference bicycle frame orientation in robot anchor frame (6D rot)."""
  command = _require_bicycle_command(env, command_name)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.ref_bike_pos(BICYCLE_TRACK_BODY),
    command.ref_bike_body_quat_w[:, command._ref_bike_index(BICYCLE_TRACK_BODY)],
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def bicycle_sim_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Simulated bicycle frame orientation in robot anchor frame (6D rot)."""
  command = _require_bicycle_command(env, command_name)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.sim_bike_pos(BICYCLE_TRACK_BODY),
    command.sim_bike_body_quat_w[:, command._sim_bike_index(BICYCLE_TRACK_BODY)],
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Privileged foot / pedal geometry
# ---------------------------------------------------------------------------


def foot_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim ankle L/R positions in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_pos_b(command, _sim_foot_pos_w(command)))


def pedal_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim pedal L/R positions in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_pos_b(command, _sim_pedal_pos_w(command)))


def foot_pos_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref ankle L/R positions in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_pos_b(command, _ref_foot_pos_w(command)))


def pedal_pos_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref pedal L/R positions in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  return _flat(_pos_b(command, _ref_pedal_pos_w(command)))


def foot_pedal_z_ref(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref world-Z of feet L/R then pedals L/R. (B, 4)."""
  command = _require_bicycle_command(env, command_name)
  fz = _ref_foot_pos_w(command)[..., 2]
  pz = _ref_pedal_pos_w(command)[..., 2]
  return torch.cat([fz, pz], dim=-1)


def foot_pedal_delta_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim (foot − pedal) L/R in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  delta_w = _sim_foot_pos_w(command) - _sim_pedal_pos_w(command)
  return _flat(_vel_b(command, delta_w))  # rotate-only (pure vector)


def foot_pedal_delta_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref (foot − pedal) L/R in robot anchor frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  delta_w = _ref_foot_pos_w(command) - _ref_pedal_pos_w(command)
  return _flat(_vel_b(command, delta_w))


def foot_pedal_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim foot L/R + pedal L/R linear velocities in robot frame. (B, 12)."""
  command = _require_bicycle_command(env, command_name)
  return torch.cat(
    [
      _flat(_vel_b(command, _sim_foot_lin_vel_w(command))),
      _flat(_vel_b(command, _sim_pedal_lin_vel_w(command))),
    ],
    dim=-1,
  )


def foot_pedal_lin_vel_ref_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref foot L/R + pedal L/R linear velocities in robot frame. (B, 12)."""
  command = _require_bicycle_command(env, command_name)
  return torch.cat(
    [
      _flat(_vel_b(command, _ref_foot_lin_vel_w(command))),
      _flat(_vel_b(command, _ref_pedal_lin_vel_w(command))),
    ],
    dim=-1,
  )


def foot_pos_err_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(foot_sim − foot_ref) L/R in robot frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  err_w = _sim_foot_pos_w(command) - _ref_foot_pos_w(command)
  return _flat(_vel_b(command, err_w))


def pedal_pos_err_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(pedal_sim − pedal_ref) L/R in robot frame. (B, 6)."""
  command = _require_bicycle_command(env, command_name)
  err_w = _sim_pedal_pos_w(command) - _ref_pedal_pos_w(command)
  return _flat(_vel_b(command, err_w))


# ---------------------------------------------------------------------------
# Contact privilege + bike joints
# ---------------------------------------------------------------------------


def bicycle_ref_contact_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference contact gates for handle/saddle/pedal pairs. (B, 5)."""
  command = _require_bicycle_command(env, command_name)
  mask = command.ref_contact_mask
  idxs = [BICYCLE_CONTACT_PAIR_KEYS.index(k) for k in CONTACT_PRIV_KEYS]
  return mask[:, idxs].float()


def bicycle_contact_force_mag(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim contact force magnitudes for handle/saddle/pedal. (B, 5)."""
  del command_name  # sensors are fixed names from BICYCLE_CONTACT_PAIRS
  parts: list[torch.Tensor] = []
  for key in CONTACT_PRIV_KEYS:
    pair = next(p for p in BICYCLE_CONTACT_PAIRS if p.key == key)
    parts.append(_contact_force_mag(env, pair.sensor_name))
  return torch.stack(parts, dim=-1)


def bicycle_pedal_net_force_local(
  env: ManagerBasedRlEnv,
  command_name: str,
  left_sensor_name: str = "contact_pedal_left_netforce",
  right_sensor_name: str = "contact_pedal_right_netforce",
  force_scale_n: float = 100.0,
) -> torch.Tensor:
  """Foot-applied net force on each pedal in that pedal's local frame. (B, 6).

  Order: ``left Fx,Fy,Fz, right Fx,Fy,Fz``.  The contact sensor normalizes its
  direction from the primary sole strips toward the secondary pedal, so its
  world-frame vector is already the force applied by the foot to the pedal.
  Summing the five foot2--6 strip vectors yields one net vector per foot.
  Dividing by ``force_scale_n`` keeps the policy input near unit scale while preserving
  both magnitude and direction without a redundant magnitude channel.
  """
  if force_scale_n <= 0.0:
    raise ValueError("force_scale_n must be positive")
  command = _require_bicycle_command(env, command_name)
  result: list[torch.Tensor] = []
  for sensor_name, pedal_body in (
    (left_sensor_name, "pedal_left"),
    (right_sensor_name, "pedal_right"),
  ):
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None:
      raise RuntimeError(f"Sensor {sensor_name!r} must provide force")
    force_on_pedal_w = force.reshape(force.shape[0], -1, 3).sum(dim=1)
    pedal_i = command._sim_bike_index(pedal_body)
    pedal_q_w = command.sim_bike_body_quat_w[:, pedal_i]
    result.append(
      quat_apply_inverse(pedal_q_w, force_on_pedal_w) / float(force_scale_n)
    )
  return torch.cat(result, dim=-1)


def bicycle_joint_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Sim bike hinge state. (B, 8).

  Order: crank, steer, pedal_l, pedal_r — each as (q, qdot).
  Scalar joint angles/vels (not robot-anchor frame).
  """
  command = _require_bicycle_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None
  crank_ids, _ = bicycle.find_joints("crank_spin")
  steer_ids, _ = bicycle.find_joints("steer")
  pedal_l_ids, _ = bicycle.find_joints("pedal_l_spin")
  pedal_r_ids, _ = bicycle.find_joints("pedal_r_spin")
  jp = bicycle.data.joint_pos
  jv = bicycle.data.joint_vel
  return torch.stack(
    [
      jp[:, crank_ids[0]],
      jv[:, crank_ids[0]],
      jp[:, steer_ids[0]],
      jv[:, steer_ids[0]],
      jp[:, pedal_l_ids[0]],
      jv[:, pedal_l_ids[0]],
      jp[:, pedal_r_ids[0]],
      jv[:, pedal_r_ids[0]],
    ],
    dim=-1,
  )


def bicycle_joint_state_ref(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ref bike hinge state from motion bike_qpos/qvel. (B, 8).

  Same order as ``bicycle_joint_state``: crank, steer, pedal_l, pedal_r (q, qdot).
  Joint layout in bike_qpos[7:]: crank, pedal_r, pedal_l, rear, steer, front.
  """
  command = _require_bicycle_command(env, command_name)
  assert command.motion.bike_qpos is not None
  assert command.motion.bike_qvel is not None
  q = command.motion.bike_qpos[command.time_steps]
  _, joint_vel = command.reference_bicycle_velocities(command.time_steps)
  # qpos: [root7 | crank, pedal_r, pedal_l, rear, steer, front]
  crank_q, pedal_r_q, pedal_l_q, steer_q = q[:, 7], q[:, 8], q[:, 9], q[:, 11]
  crank_v, pedal_r_v, pedal_l_v, steer_v = (
    joint_vel[:, 0],
    joint_vel[:, 1],
    joint_vel[:, 2],
    joint_vel[:, 4],
  )
  return torch.stack(
    [
      crank_q,
      crank_v,
      steer_q,
      steer_v,
      pedal_l_q,
      pedal_l_v,
      pedal_r_q,
      pedal_r_v,
    ],
    dim=-1,
  )


def bicycle_joint_state_legacy4(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Legacy Hist9-Rich: crank + steer only. (B, 4)."""
  return bicycle_joint_state(env, command_name)[:, :4]


def bicycle_joint_state_ref_legacy4(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Legacy Hist9-Rich ref: crank + steer only. (B, 4)."""
  return bicycle_joint_state_ref(env, command_name)[:, :4]


def raycast_depth_grid(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Normalized ray-hit distances from a :class:`RayCastSensor` grid.

  Returns shape ``(num_envs, num_rays)`` with values in ``[0, 1]``:
  0 = hit at ray origin, 1 = miss or at ``max_distance``.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  max_dist = sensor.cfg.max_distance
  distances = sensor.data.distances
  miss_mask = distances < 0
  normalized = torch.clamp(distances, min=0.0, max=max_dist) / max_dist
  return torch.where(miss_mask, torch.ones_like(normalized), normalized)
