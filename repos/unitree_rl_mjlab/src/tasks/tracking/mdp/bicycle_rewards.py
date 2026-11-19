"""Rewards for bicycle object tracking and end-effector contact."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
)

from .bicycle_data import (
  BICYCLE_CONTACT_PAIR_KEYS,
  BICYCLE_CONTACT_PAIRS,
  BICYCLE_EEF_CONTACT_PAIR_WEIGHTS,
  BICYCLE_TRACK_BODY,
  FOOT_SOLE_CONTACT_LOCAL,
  FOOT_SOLE_HEEL_TOE_MID_S0,
  FOOT_SOLE_HEEL_TOE_MID_S1,
  FOOT_SOLE_HEEL_X_LOCAL,
  FOOT_SOLE_TOE_X_LOCAL,
  T084_PEDAL_ON_TIME_S,
)

_PAIR_BY_KEY = {pair.key: (idx, pair) for idx, pair in enumerate(BICYCLE_CONTACT_PAIRS)}


def _bike_point_w(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  local_pos: tuple[float, float, float] | None,
) -> torch.Tensor:
  """World point = body origin (+ local offset if given)."""
  if local_pos is None:
    return body_pos_w
  offset = torch.tensor(local_pos, device=body_pos_w.device, dtype=body_pos_w.dtype)
  return body_pos_w + quat_apply(body_quat_w, offset.expand(body_pos_w.shape[0], -1))


def _pair_target_local(
  env: "ManagerBasedRlEnv", pair
) -> tuple[float, float, float] | None:
  if pair.key == "saddle":
    override = getattr(env, "_bicycle_saddle_contact_local", None)
    if override is not None:
      return override
  return pair.target_local_pos
from .commands import MotionCommand

_PEDAL_PAIR_INDICES: tuple[int, ...] = tuple(
  BICYCLE_CONTACT_PAIR_KEYS.index(k) for k in ("pedal_left", "pedal_right")
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _require_bicycle_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command.bicycle is None:
    raise RuntimeError(f"{command_name} has no bicycle entity configured")
  return command


def bicycle_pos_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_name: str = BICYCLE_TRACK_BODY,
) -> torch.Tensor:
  """Track simulated bicycle body position to motion reference."""
  command = _require_bicycle_command(env, command_name)
  error = torch.sum(
    torch.square(command.ref_bike_pos(body_name) - command.sim_bike_pos(body_name)),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def bicycle_ori_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_name: str = BICYCLE_TRACK_BODY,
) -> torch.Tensor:
  """Track simulated bicycle body orientation to motion reference."""
  command = _require_bicycle_command(env, command_name)
  ref_i = command._ref_bike_index(body_name)
  sim_i = command._sim_bike_index(body_name)
  error = quat_error_magnitude(
    command.ref_bike_body_quat_w[:, ref_i],
    command.sim_bike_body_quat_w[:, sim_i],
  ) ** 2
  return torch.exp(-error / std**2)


def _pedal_foot_above_gate(
  ee_pos: torch.Tensor,
  pedal_pos: torch.Tensor,
  *,
  z_margin: float,
  z_soft: float,
) -> torch.Tensor:
  """Soft gate in [0, 1]: 1 when foot Z is above pedal Z by ``z_margin``.

  ``dz = foot_z - pedal_z``. Full credit for ``dz >= z_margin``, zero for
  ``dz <= z_margin - z_soft`` (or hard step if ``z_soft <= 0``).
  """
  dz = ee_pos[:, 2] - pedal_pos[:, 2]
  if z_soft <= 0.0:
    return (dz >= z_margin).float()
  return ((dz - (z_margin - z_soft)) / z_soft).clamp(0.0, 1.0)


def _pedal_contact_surface_data(
  sensor: ContactSensor,
  pedal_center_w: torch.Tensor,
  pedal_quat_w: torch.Tensor,
  *,
  top_min_local_z: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return top force, top gate, and exclusively-non-top gate for one pedal.

  Contact positions are transformed into the pedal body's local frame, relative
  to the collision-box center.  The pedal box half-height is 0.015 m; using a
  positive threshold selects its top face and rejects bottom/low-side contacts.
  """
  found = sensor.data.found
  force = sensor.data.force
  pos = sensor.data.pos
  if found is None or force is None or pos is None:
    raise RuntimeError(
      f"Sensor {sensor.cfg.name!r} must provide found, force, and pos "
      "for pedal surface classification"
    )
  if found.ndim == 1:
    found = found.unsqueeze(-1)
  if force.ndim == 2:
    force = force.unsqueeze(1)
  if pos.ndim == 2:
    pos = pos.unsqueeze(1)

  n_contacts = pos.shape[1]
  rel_w = pos - pedal_center_w.unsqueeze(1)
  pedal_quat = pedal_quat_w.unsqueeze(1).expand(-1, n_contacts, -1)
  rel_local = quat_apply_inverse(
    pedal_quat.reshape(-1, 4), rel_w.reshape(-1, 3)
  ).reshape(rel_w.shape)

  contact = found > 0
  top = contact & (rel_local[..., 2] >= float(top_min_local_z))
  non_top = contact & ~top
  force_mag = torch.norm(force, dim=-1)
  top_force = torch.where(top, force_mag, torch.zeros_like(force_mag)).amax(dim=-1)
  top_gate = top.any(dim=-1)
  # A correctly seated foot can touch the top and an upper side with different
  # capsule geoms in the same step.  Penalize only contacts with no top support.
  exclusively_non_top = non_top.any(dim=-1) & ~top_gate
  return top_force, top_gate.float(), exclusively_non_top.float()


def _handle_inner_plate_gate(
  sensor: ContactSensor,
  *,
  min_plates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Force mag + nesting gate for ㄷ-hand cavity-facing face pads.

  Handle sensors match ``*_hand_plate_{back,left,right}_inner_collision``
  (thin pads on the ONE face toward the ㄷ cavity). Requiring ``min_plates``
  distinct pads in contact approximates a nested grip.
  """
  force = sensor.data.force
  found = sensor.data.found
  assert force is not None and found is not None
  n_slots = max(int(sensor.cfg.num_slots), 1)
  if found.ndim == 1:
    found = found.unsqueeze(-1)
  n_total = found.shape[-1]
  n_prim = max(n_total // n_slots, 1)
  B = found.shape[0]
  found_ps = found.reshape(B, n_prim, n_slots)
  plate_hit = (found_ps > 0).any(dim=-1)
  gate = (plate_hit.sum(dim=-1) >= int(min_plates)).float()

  if force.ndim == 3:
    frc = force.reshape(B, n_prim, n_slots, 3)
    frc_mag = torch.norm(frc, dim=-1).amax(dim=-1)
    frc_mag = (frc_mag * plate_hit.float()).amax(dim=-1)
  else:
    frc = force.reshape(B, n_prim, n_slots)
    frc_mag = frc.abs().amax(dim=-1)
    frc_mag = (frc_mag * plate_hit.float()).amax(dim=-1)
  return frc_mag, gate


def _contact_pair_signals(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  pair_key: str,
  *,
  pos_sigma: float,
  pos_tolerance: float,
  frc_sigma: float,
  frc_thres: float,
  pedal_z_margin: float = 0.0,
  pedal_z_soft: float = 0.02,
  handle_min_inner_plates: int = 1,
  position_target: Literal["reference", "simulated"] = "reference",
  pedal_top_contact: bool = False,
  pedal_top_min_local_z: float = 0.0075,
  pedal_sole_contact: bool = False,
  pedal_forefoot_filter: bool = True,
  force_shaping: Literal["exp_threshold", "tanh"] = "exp_threshold",
  force_scale: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-pair approach, force-bonus, and reference-schedule mask.

  Returns ``(pos_rew, force_term, schedule_mask, valid_contact)``.
  ``force_term`` is already gated by actual sensor contact / nesting / top-face.
  """
  if pair_key not in _PAIR_BY_KEY:
    raise ValueError(f"unknown bicycle contact pair {pair_key!r}")
  pair_idx, pair = _PAIR_BY_KEY[pair_key]

  ee_pos = command.robot_body_pos_by_name(pair.ee_body)
  if pair.key in ("pedal_left", "pedal_right") and pedal_sole_contact:
    ee_pos = _bike_point_w(
      ee_pos,
      command.robot_body_quat_by_name(pair.ee_body),
      FOOT_SOLE_CONTACT_LOCAL,
    )
  if pair.target_mode == "ground":
    ground_z = command._env.scene.env_origins[:, 2]
    pos_err = (ee_pos[:, 2] - ground_z).abs()
  else:
    local = _pair_target_local(env, pair)
    if position_target == "simulated":
      body_pos = command.sim_bike_body_pos_w[:, command._sim_bike_index(pair.bike_body)]
      body_quat = command.sim_bike_body_quat_w[:, command._sim_bike_index(pair.bike_body)]
    else:
      body_pos = command.ref_bike_body_pos_w[:, command._ref_bike_index(pair.bike_body)]
      body_quat = command.ref_bike_body_quat_w[:, command._ref_bike_index(pair.bike_body)]
    tgt_pos = _bike_point_w(body_pos, body_quat, local)
    pos_err = torch.norm(ee_pos - tgt_pos, dim=-1)
  pos_err = (pos_err - pos_tolerance).clamp_min(0.0)
  pos_rew = torch.exp(-pos_err / pos_sigma)

  sensor: ContactSensor = env.scene[pair.sensor_name]
  pedal_top_gate = None
  nest_gate = None
  if pair.key.startswith("handle") and handle_min_inner_plates > 1:
    frc_mag, nest_gate = _handle_inner_plate_gate(
      sensor, min_plates=handle_min_inner_plates
    )
  elif pair.key in ("pedal_left", "pedal_right") and pedal_top_contact:
    sim_i = command._sim_bike_index(pair.bike_body)
    sim_pedal_quat = command.sim_bike_body_quat_w[:, sim_i]
    sim_pedal = _bike_point_w(
      command.sim_bike_body_pos_w[:, sim_i],
      sim_pedal_quat,
      _pair_target_local(env, pair),
    )
    frc_mag, pedal_top_gate, _ = _pedal_contact_surface_data(
      sensor,
      sim_pedal,
      sim_pedal_quat,
      top_min_local_z=pedal_top_min_local_z,
    )
  elif (
    pair.key in ("pedal_left", "pedal_right")
    and pedal_sole_contact
    and pedal_forefoot_filter
  ):
    found = sensor.data.found
    force = sensor.data.force
    pos = sensor.data.pos
    if found is None or force is None or pos is None:
      raise RuntimeError(
        f"Sensor {sensor.cfg.name!r} needs found/force/pos for forefoot gating"
      )
    foot_quat = command.robot_body_quat_by_name(pair.ee_body)
    foot_pos = command.robot_body_pos_by_name(pair.ee_body)
    rel_w = pos - foot_pos.unsqueeze(1)
    q = foot_quat.unsqueeze(1).expand(-1, rel_w.shape[1], -1)
    rel_b = quat_apply_inverse(q.reshape(-1, 4), rel_w.reshape(-1, 3)).reshape(
      rel_w.shape
    )
    span = FOOT_SOLE_TOE_X_LOCAL - FOOT_SOLE_HEEL_X_LOCAL
    mid_x0 = FOOT_SOLE_HEEL_X_LOCAL + FOOT_SOLE_HEEL_TOE_MID_S0 * span
    mid_x1 = FOOT_SOLE_HEEL_X_LOCAL + FOOT_SOLE_HEEL_TOE_MID_S1 * span
    mid_hit = (found > 0) & (rel_b[..., 0] >= mid_x0) & (rel_b[..., 0] <= mid_x1)
    force_mag = torch.norm(force, dim=-1)
    frc_mag = torch.where(
      mid_hit, force_mag, torch.zeros_like(force_mag)
    ).amax(dim=-1)
    pedal_top_gate = mid_hit.any(dim=-1).float()
  else:
    force = sensor.data.force
    assert force is not None
    if force.ndim == 3:
      frc_mag = torch.norm(force, dim=-1).amax(dim=-1)
    else:
      frc_mag = torch.norm(force, dim=-1)
  if force_shaping == "tanh":
    if force_scale <= 0.0:
      raise ValueError("force_scale must be positive")
    frc_sat = torch.tanh(frc_mag / float(force_scale))
  elif force_shaping == "exp_threshold":
    frc_sat = torch.exp((frc_mag - frc_thres) / frc_sigma).clamp_max(1.0)
  else:
    raise ValueError(f"unknown force_shaping {force_shaping!r}")
  found = sensor.data.found
  assert found is not None
  contact_gate = found.reshape(found.shape[0], -1).any(dim=-1).float()
  if nest_gate is not None:
    contact_gate = nest_gate
  elif pedal_top_gate is not None:
    contact_gate = pedal_top_gate
  force_term = frc_sat * contact_gate

  if (
    pair.key in ("pedal_left", "pedal_right")
    and not pedal_top_contact
    and not pedal_sole_contact
  ):
    local = _pair_target_local(env, pair)
    sim_i = command._sim_bike_index(pair.bike_body)
    sim_pedal = _bike_point_w(
      command.sim_bike_body_pos_w[:, sim_i],
      command.sim_bike_body_quat_w[:, sim_i],
      local,
    )
    above = _pedal_foot_above_gate(
      ee_pos,
      sim_pedal,
      z_margin=pedal_z_margin,
      z_soft=pedal_z_soft,
    )
    pos_rew = pos_rew * above
    force_term = force_term * above

  schedule = command.ref_contact_mask[:, pair_idx].float()
  return pos_rew, force_term, schedule, contact_gate


def hdmi_eef_contact_reward(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  *,
  pos_sigma: float,
  pos_tolerance: float,
  frc_sigma: float,
  frc_thres: float,
  gain: float,
  pedal_z_margin: float = 0.0,
  pedal_z_soft: float = 0.02,
  handle_min_inner_plates: int = 1,
  pair_weights: dict[str, float] | None = None,
  position_target: Literal["reference", "simulated"] = "reference",
  require_contact: bool = False,
  pedal_top_contact: bool = False,
  pedal_top_min_local_z: float = 0.0075,
  pedal_sole_contact: bool = False,
  pedal_forefoot_filter: bool = True,
) -> torch.Tensor:
  """HDMI unified interaction reward (arXiv:2509.16757 eqs. 1–2).

  For each EE pair i with reference contact gate ``c_i = 1``:

    R_i = exp(-||p_eef - p_tgt||_2 / σ_pos)
          * min(exp((||F_i||_2 - F_thres) / σ_frc), 1)

  R_interaction = Σ_i w_i R_i c_i / Σ_i w_i c_i    (0 if denom = 0)

  Default weights: handle L/R 20+20 (=40), saddle 20, pedal L/R 20+20 (=40).
  Left/right pedals (and hands) are separate pairs with independent ``c_i``.

  Pedal pairs (default): gated so contact only scores when the foot is
  above the pedal (``foot_z >= pedal_z + pedal_z_margin``).

  ``pedal_sole_contact=True``: distance from thin sole-pad center (not ankle
  origin); force from sole-only sensors.  With ``pedal_top_contact=False``
  (current crank-progress tasks), any sole↔pedal-box contact counts — pedal
  faces are not classified, because the pedal rotates.

  Handle pairs (ㄷ-hand): force only counts when ``handle_min_inner_plates``
  of {back, left, right} inner plates contact the bar (nesting gate).
  Use 2 for dhand, 1 for rubber.

  Before attach frames: ``c_i = 0`` → no contact reward for that pair.
  During attach: approaching raises the position term; once force reaches
  ``F_thres`` the force term saturates at 1 (not unbounded growth).
  After detach: ``c_i = 0`` again.
  """
  weights = pair_weights if pair_weights is not None else BICYCLE_EEF_CONTACT_PAIR_WEIGHTS
  pair_rewards: list[torch.Tensor] = []
  masks: list[torch.Tensor] = []
  weight_list: list[float] = []

  for pair in BICYCLE_CONTACT_PAIRS:
    weight_list.append(float(weights.get(pair.key, 0.0)))
    pos_rew, force_term, schedule, _ = _contact_pair_signals(
      env,
      command,
      pair.key,
      pos_sigma=pos_sigma,
      pos_tolerance=pos_tolerance,
      frc_sigma=frc_sigma,
      frc_thres=frc_thres,
      pedal_z_margin=pedal_z_margin,
      pedal_z_soft=pedal_z_soft,
      handle_min_inner_plates=handle_min_inner_plates,
      position_target=position_target,
      pedal_top_contact=pedal_top_contact,
      pedal_top_min_local_z=pedal_top_min_local_z,
      pedal_sole_contact=pedal_sole_contact,
      pedal_forefoot_filter=pedal_forefoot_filter,
    )
    # Legacy product.  ``require_contact=False`` lets distance score without a
    # sensor hit by treating the force factor as 1.
    frc_rew = force_term if require_contact else torch.where(
      force_term > 0.0, force_term, torch.ones_like(force_term)
    )
    pair_rewards.append(pos_rew * frc_rew * gain)
    masks.append(schedule)

  stacked = torch.stack(pair_rewards, dim=-1)  # (B, N)
  mask = torch.stack(masks, dim=-1)  # (B, N)
  w = torch.tensor(weight_list, device=stacked.device, dtype=stacked.dtype)
  numbered = (stacked * mask * w).sum(dim=-1)
  denom = (mask * w).sum(dim=-1)
  reward = numbered / denom.clamp_min(1.0e-8)
  return torch.where(denom > 0.0, reward, torch.zeros_like(denom))


def bicycle_eef_contact_exp(
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
  position_target: Literal["reference", "simulated"] = "reference",
  require_contact: bool = False,
  pedal_top_contact: bool = False,
  pedal_top_min_local_z: float = 0.0075,
  pedal_sole_contact: bool = False,
  pedal_forefoot_filter: bool = True,
) -> torch.Tensor:
  """HDMI-style EE contact reward (see ``hdmi_eef_contact_reward``)."""
  command = _require_bicycle_command(env, command_name)
  return hdmi_eef_contact_reward(
    env,
    command,
    pos_sigma=pos_sigma,
    pos_tolerance=pos_tolerance,
    frc_sigma=frc_sigma,
    frc_thres=frc_thres,
    gain=gain,
    pedal_z_margin=pedal_z_margin,
    pedal_z_soft=pedal_z_soft,
    handle_min_inner_plates=handle_min_inner_plates,
    pair_weights=pair_weights,
    position_target=position_target,
    require_contact=require_contact,
    pedal_top_contact=pedal_top_contact,
    pedal_top_min_local_z=pedal_top_min_local_z,
    pedal_sole_contact=pedal_sole_contact,
    pedal_forefoot_filter=pedal_forefoot_filter,
  )


def bicycle_contact_pair_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  pair_key: str,
  pos_sigma: float = 0.3,
  pos_tolerance: float = 0.02,
  frc_sigma: float = 40.0,
  frc_thres: float = 20.0,
  approach_weight: float = 0.5,
  force_weight: float = 0.5,
  pedal_z_margin: float = 0.0,
  pedal_z_soft: float = 0.02,
  handle_min_inner_plates: int = 1,
  position_target: Literal["reference", "simulated"] = "reference",
  pedal_top_contact: bool = False,
  pedal_top_min_local_z: float = 0.0075,
  pedal_sole_contact: bool = False,
  pedal_forefoot_filter: bool = True,
  force_shaping: Literal["exp_threshold", "tanh"] = "exp_threshold",
  force_scale: float = 20.0,
  precontact_frames: int = 0,
  exclusive_distance_force: bool = False,
) -> torch.Tensor:
  """Schedule-gated contact reward for one bicycle interaction pair.

  Active only while the reference contact mask for ``pair_key`` is 1:

    R = c_i * (w_approach * pos + w_force * contact_force)

  ``pos`` guides the end-effector toward the target.  ``contact_force`` is
  extra credit for an actual sensor contact, saturating near ``frc_thres``.
  The two weights default to 0.5 so a fully successful pair scores 1.
  """
  command = _require_bicycle_command(env, command_name)
  pos_rew, force_term, schedule, valid_contact = _contact_pair_signals(
    env,
    command,
    pair_key,
    pos_sigma=pos_sigma,
    pos_tolerance=pos_tolerance,
    frc_sigma=frc_sigma,
    frc_thres=frc_thres,
    pedal_z_margin=pedal_z_margin,
    pedal_z_soft=pedal_z_soft,
    handle_min_inner_plates=handle_min_inner_plates,
    position_target=position_target,
    pedal_top_contact=pedal_top_contact,
    pedal_top_min_local_z=pedal_top_min_local_z,
    pedal_sole_contact=pedal_sole_contact,
    pedal_forefoot_filter=pedal_forefoot_filter,
    force_shaping=force_shaping,
    force_scale=force_scale,
  )
  approach_gate = schedule.bool()
  if precontact_frames > 0:
    t_max = max(int(command.motion.time_step_total) - 1, 0)
    future = torch.stack(
      [
        command._ref_contact_mask[
          torch.clamp(command.time_steps + step, max=t_max),
          _PAIR_BY_KEY[pair_key][0],
        ]
        for step in range(1, int(precontact_frames) + 1)
      ],
      dim=-1,
    )
    approach_gate = approach_gate | future.bool().any(dim=-1)
  if exclusive_distance_force:
    approach_gate = approach_gate & ~valid_contact.bool()
  return (
    approach_weight * approach_gate.float() * pos_rew
    + force_weight * schedule * force_term
  )


def bicycle_roll_dynamics_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  lean_std: float = 0.2,
  roll_rate_std: float = 1.0,
) -> torch.Tensor:
  """Track reference bicycle lean and roll rate without constraining global yaw."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  sim_i = command._sim_bike_index("frame")
  ref_i = command._ref_bike_index("frame")
  sim_q = command.sim_bike_body_quat_w[:, sim_i]
  ref_q = command.ref_bike_body_quat_w[:, ref_i]
  gravity = torch.zeros(env.num_envs, 3, device=env.device, dtype=sim_q.dtype)
  gravity[:, 2] = -1.0
  sim_lean = quat_apply_inverse(sim_q, gravity)[:, 1]
  ref_lean = quat_apply_inverse(ref_q, gravity)[:, 1]
  x_axis = torch.zeros_like(gravity)
  x_axis[:, 0] = 1.0
  sim_forward = quat_apply(sim_q, x_axis)
  sim_rate = (command.bicycle.data.root_link_ang_vel_w * sim_forward).sum(dim=-1)
  root_ref_vel, _ = command.reference_bicycle_velocities(command.time_steps)
  ref_forward = quat_apply(ref_q, x_axis)
  ref_rate = (root_ref_vel[:, 3:] * ref_forward).sum(dim=-1)
  error = ((sim_lean - ref_lean) / lean_std) ** 2
  error = error + ((sim_rate - ref_rate) / roll_rate_std) ** 2
  return torch.exp(-error)


def bicycle_forward_speed_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
) -> torch.Tensor:
  """Track bicycle forward speed while allowing global trajectory correction."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  sim_i = command._sim_bike_index("frame")
  ref_i = command._ref_bike_index("frame")
  x_axis = torch.zeros(env.num_envs, 3, device=env.device)
  x_axis[:, 0] = 1.0
  sim_forward = quat_apply(command.sim_bike_body_quat_w[:, sim_i], x_axis)
  ref_forward = quat_apply(command.ref_bike_body_quat_w[:, ref_i], x_axis)
  sim_speed = (command.bicycle.data.root_link_lin_vel_w * sim_forward).sum(dim=-1)
  ref_root_vel, _ = command.reference_bicycle_velocities(command.time_steps)
  ref_speed = (ref_root_vel[:, :3] * ref_forward).sum(dim=-1)
  return torch.exp(-torch.square(sim_speed - ref_speed) / std**2)


def bicycle_crank_cadence_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 2.0,
) -> torch.Tensor:
  """Track physical crank angular velocity rather than its global trajectory."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  crank_ids, _ = command.bicycle.find_joints("crank_spin")
  sim_rate = command.bicycle.data.joint_vel[:, crank_ids[0]]
  _, ref_joint_vel = command.reference_bicycle_velocities(command.time_steps)
  ref_rate = ref_joint_vel[:, 0]
  return torch.exp(-torch.square(sim_rate - ref_rate) / std**2)


def _pedal_schedule_active(command: MotionCommand) -> torch.Tensor:
  """Whether either reference pedal-contact channel is active."""
  return command.ref_contact_mask[:, _PEDAL_PAIR_INDICES].any(dim=-1)


class BicycleCrankWindowProgress:
  """Bounded forward-crank progress over a short rolling time window.

  Averaging signed crank velocity over several policy steps makes a forward
  impulse followed by a backward reset score near zero, while continuous
  forward rotation tracks the positive reference cadence.  The term is active
  only while either pedal's reference contact schedule is on.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._window_steps = int(cfg.params.get("window_steps", 10))
    if self._window_steps <= 0:
      raise ValueError("window_steps must be positive")
    self._rates = torch.zeros(
      env.num_envs, self._window_steps, device=env.device, dtype=torch.float32
    )
    self._counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._cursor = 0

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._rates[env_ids] = 0.0
    self._counts[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    window_steps: int = 10,
    forward_sign: float = 1.0,
    min_target_rate: float = 0.5,
  ) -> torch.Tensor:
    if int(window_steps) != self._window_steps:
      raise ValueError(
        f"Configured window_steps changed from {self._window_steps} to {window_steps}"
      )
    command = _require_bicycle_command(env, command_name)
    assert command.bicycle is not None
    crank_ids, _ = command.bicycle.find_joints("crank_spin")
    signed_rate = (
      float(forward_sign) * command.bicycle.data.joint_vel[:, crank_ids[0]]
    )
    active = _pedal_schedule_active(command)

    # Clear stale samples whenever an environment leaves the pedaling window.
    self._rates[~active] = 0.0
    self._rates[:, self._cursor] = torch.where(
      active, signed_rate, torch.zeros_like(signed_rate)
    )
    self._counts = torch.where(
      active,
      torch.clamp(self._counts + 1, max=self._window_steps),
      torch.zeros_like(self._counts),
    )
    self._cursor = (self._cursor + 1) % self._window_steps

    mean_rate = self._rates.sum(dim=-1) / self._counts.clamp_min(1).float()
    _, ref_joint_vel = command.reference_bicycle_velocities(command.time_steps)
    target_rate = (
      float(forward_sign) * ref_joint_vel[:, 0]
    ).clamp_min(float(min_target_rate))
    progress = (mean_rate / target_rate).clamp(0.0, 1.0)
    return progress * active.float()


def bicycle_crank_backward_rate_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  forward_sign: float = 1.0,
  tolerance: float = 0.15,
  rate_scale: float = 2.0,
) -> torch.Tensor:
  """Bounded backward crank-rate cost during scheduled pedaling."""
  if rate_scale <= 0.0:
    raise ValueError("rate_scale must be positive")
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  crank_ids, _ = command.bicycle.find_joints("crank_spin")
  signed_rate = float(forward_sign) * command.bicycle.data.joint_vel[:, crank_ids[0]]
  backward = ((-signed_rate - float(tolerance)).clamp_min(0.0) / rate_scale).clamp(
    max=1.0
  )
  return backward * _pedal_schedule_active(command).float()


def contact_force_history_violation_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
  max_count: int = 1,
) -> torch.Tensor:
  """Count policy substeps containing a forbidden contact, with a hard cap."""
  if max_count <= 0:
    raise ValueError("max_count must be positive")
  sensor: ContactSensor = env.scene[sensor_name]
  history = sensor.data.force_history
  if history is not None:
    force_mag = torch.norm(history, dim=-1)  # (B, primary, history)
    hits = (force_mag > float(force_threshold)).any(dim=1).sum(dim=-1)
  else:
    force = sensor.data.force
    if force is None:
      raise RuntimeError(f"Sensor {sensor_name!r} must provide force")
    hits = (torch.norm(force, dim=-1) > float(force_threshold)).any(dim=1).long()
  return hits.clamp(max=int(max_count)).float()


def bicycle_post_schedule_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  pair_key: str,
  sensor_name: str,
  grace_frames: int = 5,
) -> torch.Tensor:
  """Penalize contact only after a pair's scheduled falling edge.

  A cumulative reference-mask lookup distinguishes post-detach frames from
  pre-attach frames and remains correct for RSI resets into the end of a clip.
  """
  if grace_frames < 0:
    raise ValueError("grace_frames must be non-negative")
  command = _require_bicycle_command(env, command_name)
  if pair_key not in _PAIR_BY_KEY:
    raise ValueError(f"unknown bicycle contact pair {pair_key!r}")
  pair_idx = _PAIR_BY_KEY[pair_key][0]
  mask = command._ref_contact_mask[:, pair_idx].bool()
  frame = torch.arange(mask.shape[0], device=mask.device, dtype=torch.long)
  last_on = torch.cummax(torch.where(mask, frame, -torch.ones_like(frame)), dim=0)[0]
  age = frame - last_on
  post_off = (last_on >= 0) & ~mask & (age > int(grace_frames))

  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    raise RuntimeError(f"Sensor {sensor_name!r} must provide found")
  in_contact = found.reshape(found.shape[0], -1).any(dim=-1)
  return post_off[command.time_steps].float() * in_contact.float()


class BicycleCrankDirectionReversal:
  """Emit one when the moving crank changes direction in the pedal window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._previous_direction = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.int8
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._previous_direction[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    forward_sign: float = 1.0,
    min_abs_rate: float = 0.25,
  ) -> torch.Tensor:
    command = _require_bicycle_command(env, command_name)
    assert command.bicycle is not None
    crank_ids, _ = command.bicycle.find_joints("crank_spin")
    signed_rate = float(forward_sign) * command.bicycle.data.joint_vel[:, crank_ids[0]]
    active = _pedal_schedule_active(command)
    moving = signed_rate.abs() >= float(min_abs_rate)
    direction = torch.where(
      signed_rate >= 0.0,
      torch.ones_like(self._previous_direction),
      -torch.ones_like(self._previous_direction),
    )
    reversal = (
      active
      & moving
      & (self._previous_direction != 0)
      & (direction != self._previous_direction)
    )
    self._previous_direction = torch.where(
      active & moving,
      direction,
      torch.where(active, self._previous_direction, 0),
    )
    return reversal.float()


def bicycle_pedal_underside_contact_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  top_min_local_z: float = 0.0075,
) -> torch.Tensor:
  """Return one when either foot contacts a pedal away from its top face."""
  command = _require_bicycle_command(env, command_name)
  underside: list[torch.Tensor] = []
  for pair_idx in _PEDAL_PAIR_INDICES:
    pair = BICYCLE_CONTACT_PAIRS[pair_idx]
    sensor: ContactSensor = env.scene[pair.sensor_name]
    sim_i = command._sim_bike_index(pair.bike_body)
    pedal_quat = command.sim_bike_body_quat_w[:, sim_i]
    pedal_center = _bike_point_w(
      command.sim_bike_body_pos_w[:, sim_i],
      pedal_quat,
      _pair_target_local(env, pair),
    )
    _, _, non_top_gate = _pedal_contact_surface_data(
      sensor,
      pedal_center,
      pedal_quat,
      top_min_local_z=top_min_local_z,
    )
    underside.append(non_top_gate)
  return torch.stack(underside, dim=-1).amax(dim=-1)


def bicycle_rear_wheel_no_slip_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_radius: float = 0.24,
  std: float = 0.5,
) -> torch.Tensor:
  """Reward rolling consistency ``v_forward ~= radius * rear_wheel_rate``."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  frame_i = command._sim_bike_index("frame")
  x_axis = torch.zeros(env.num_envs, 3, device=env.device)
  x_axis[:, 0] = 1.0
  forward = quat_apply(command.sim_bike_body_quat_w[:, frame_i], x_axis)
  speed = (command.bicycle.data.root_link_lin_vel_w * forward).sum(dim=-1)
  rear_ids, _ = command.bicycle.find_joints("rear_spin")
  rolling_speed = wheel_radius * command.bicycle.data.joint_vel[:, rear_ids[0]]
  return torch.exp(-torch.square(speed - rolling_speed) / std**2)


def bicycle_rear_wheel_longitudinal_slip_abs(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_radius: float = 0.24,
) -> torch.Tensor:
  """Absolute longitudinal rolling error for logging, not reward shaping."""
  command = _require_bicycle_command(env, command_name)
  assert command.bicycle is not None
  frame_i = command._sim_bike_index("frame")
  x_axis = torch.zeros(env.num_envs, 3, device=env.device)
  x_axis[:, 0] = 1.0
  forward = quat_apply(command.sim_bike_body_quat_w[:, frame_i], x_axis)
  speed = (command.bicycle.data.root_link_lin_vel_w * forward).sum(dim=-1)
  rear_ids, _ = command.bicycle.find_joints("rear_spin")
  rolling_speed = wheel_radius * command.bicycle.data.joint_vel[:, rear_ids[0]]
  return (speed - rolling_speed).abs()


def bicycle_pedal_force_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  force_scale: float = 50.0,
  force_clamp: float = 10.0,
  pedal_z_margin: float = 0.0,
  pedal_z_soft: float = 0.02,
) -> torch.Tensor:
  """Proportional foot–pedal contact force reward (gated by pedal contact flags).

  For each active pedal pair i: ``r_i = clamp(||F_i|| / force_scale, 0, force_clamp)``.
  Only counts when the foot is above the pedal (same Z gate as EE contact).
  Returns the mean over active pedal gates (0 if none active).
  """
  command = _require_bicycle_command(env, command_name)
  pair_rewards: list[torch.Tensor] = []
  masks: list[torch.Tensor] = []

  for pair_idx in _PEDAL_PAIR_INDICES:
    pair = BICYCLE_CONTACT_PAIRS[pair_idx]
    sensor: ContactSensor = env.scene[pair.sensor_name]
    force = sensor.data.force
    assert force is not None
    if force.ndim == 3:
      frc_mag = torch.norm(force, dim=-1).amax(dim=-1)
    else:
      frc_mag = torch.norm(force, dim=-1)
    ee_pos = command.robot_body_pos_by_name(pair.ee_body)
    local = _pair_target_local(env, pair)
    sim_i = command._sim_bike_index(pair.bike_body)
    pedal_pos = _bike_point_w(
      command.sim_bike_body_pos_w[:, sim_i],
      command.sim_bike_body_quat_w[:, sim_i],
      local,
    )
    z_gate = _pedal_foot_above_gate(
      ee_pos,
      pedal_pos,
      z_margin=pedal_z_margin,
      z_soft=pedal_z_soft,
    )
    pair_rewards.append((frc_mag / force_scale).clamp(max=force_clamp) * z_gate)
    masks.append(command.ref_contact_mask[:, pair_idx].float())

  stacked = torch.stack(pair_rewards, dim=-1)
  mask = torch.stack(masks, dim=-1)
  n_active = mask.sum(dim=-1)
  numbered = (stacked * mask).sum(dim=-1)
  return torch.where(n_active > 0.0, numbered / n_active, torch.zeros_like(n_active))


def bicycle_pedal_approach_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_sigma: float = 0.3,
  pre_horizon_s: float = 2.0,
) -> torch.Tensor:
  """Pre-contact guidance: reward feet approaching pedals before each pedal ON time.

  For pedal i with first ON time ``t_on`` (from T084 schedule), active when
  ``t_on - pre_horizon_s <= t < t_on``. Reward is ``exp(-||p_foot - p_pedal|| / σ)``.
  After contact gate turns on, this term is zero (contact/force rewards take over).
  Hands/saddle are unaffected.
  """
  command = _require_bicycle_command(env, command_name)
  # Motion timeline at 50 Hz (matches baked contact flags / T084 schedule).
  t = command.time_steps.float() * (1.0 / 50.0)

  pair_rewards: list[torch.Tensor] = []
  masks: list[torch.Tensor] = []
  for pair_idx in _PEDAL_PAIR_INDICES:
    pair = BICYCLE_CONTACT_PAIRS[pair_idx]
    t_on = T084_PEDAL_ON_TIME_S[pair.key]
    t_start = max(0.0, t_on - pre_horizon_s)
    guide = ((t >= t_start) & (t < t_on)).float()

    ee_pos = command.robot_body_pos_by_name(pair.ee_body)
    tgt_pos = command.ref_bike_pos(pair.bike_body)
    dist = torch.norm(ee_pos - tgt_pos, dim=-1)
    pair_rewards.append(torch.exp(-dist / pos_sigma))
    masks.append(guide)

  stacked = torch.stack(pair_rewards, dim=-1)
  mask = torch.stack(masks, dim=-1)
  n_active = mask.sum(dim=-1)
  numbered = (stacked * mask).sum(dim=-1)
  return torch.where(n_active > 0.0, numbered / n_active, torch.zeros_like(n_active))


def foot_ground_penalty_in_motion_window(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  start_time_s: float = 6.0,
  end_time_s: float = 9.5,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize foot-floor contact while the motion is in ``[start_time_s, end_time_s]``.

  Returns 1 when either foot exceeds *force_threshold* against terrain, else 0.
  Gating uses motion ``time_steps`` × env step dt (50 Hz motion timeline).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_time_s = command.time_steps.float() * env.step_dt
  time_mask = (motion_time_s >= start_time_s) & (motion_time_s <= end_time_s)

  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  if force.ndim == 3:
    frc_mag = torch.norm(force, dim=-1).amax(dim=-1)
  else:
    frc_mag = torch.norm(force, dim=-1)
  foot_contact = (frc_mag > force_threshold).float()
  return foot_contact * time_mask.float()


def bicycle_foot_ground_force_in_frame_window(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str = "contact_foot_left_ground",
  start_frame: int = 244,
  end_frame: int = 280,
  force_scale: float = 50.0,
  force_clamp: float = 10.0,
) -> torch.Tensor:
  """Proportional foot–ground contact force in a motion frame window.

  Used for t060 left-foot push-off (frames 244–280 @ 50 Hz): reward scales with
  ``clamp(||F|| / force_scale, 0, force_clamp)`` while ``start_frame <= t <= end_frame``,
  else 0. Encourages stamping the ground to drive the bike forward before the
  left pedal contact gate turns on.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  frame_mask = (command.time_steps >= start_frame) & (
    command.time_steps <= end_frame
  )

  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  if force.ndim == 3:
    frc_mag = torch.norm(force, dim=-1).amax(dim=-1)
  else:
    frc_mag = torch.norm(force, dim=-1)
  rew = (frc_mag / force_scale).clamp(max=force_clamp)
  return rew * frame_mask.float()


def bicycle_forward_accel_in_frame_window(
  env: ManagerBasedRlEnv,
  command_name: str,
  start_frame: int = 230,
  end_frame: int = 280,
  accel_scale: float = 2.0,
  accel_clamp: float = 2.0,
  body_name: str = BICYCLE_TRACK_BODY,
) -> torch.Tensor:
  """Reward positive bicycle forward acceleration in a motion frame window.

  Forward = body +X of ``body_name`` (frame). Acceleration is a finite difference
  of the projected root linear velocity across env steps. Outside the window → 0.
  """
  command = _require_bicycle_command(env, command_name)
  bicycle = command.bicycle
  assert bicycle is not None

  frame_mask = (command.time_steps >= start_frame) & (
    command.time_steps <= end_frame
  )

  body_ids, _ = bicycle.find_bodies(body_name)
  body_id = body_ids[0]
  quat = bicycle.data.body_link_quat_w[:, body_id]
  vel_w = bicycle.data.root_link_lin_vel_w
  x_axis = torch.zeros(env.num_envs, 3, device=env.device, dtype=vel_w.dtype)
  x_axis[:, 0] = 1.0
  forward = quat_apply(quat, x_axis)
  v_fwd = (vel_w * forward).sum(dim=-1)

  prev_key = "_bicycle_prev_v_fwd"
  prev = getattr(env, prev_key, None)
  if prev is None or prev.shape != v_fwd.shape:
    prev = v_fwd.detach().clone()
  accel = (v_fwd - prev) / max(float(env.step_dt), 1e-6)
  setattr(env, prev_key, v_fwd.detach().clone())

  # Only reward forward (pushing the bike ahead); clamp for scale stability.
  rew = (accel / accel_scale).clamp(min=0.0, max=accel_clamp)
  return rew * frame_mask.float()
