"""Opt-in physical-state initialization (PSI) for dynamic bicycle tracking.

This module intentionally does not change :class:`MotionCommand`.  A task must
replace its motion command config with :class:`BicyclePSIMotionCommandCfg` to
enable the buffer.  The implementation is inspired by InterMimic's physical
buffer, but acceptance is based on a future validation window and explicit
bicycle/contact gates rather than a scalar reward threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .bicycle_data import BICYCLE_CONTACT_PAIRS, BICYCLE_SPLIT_CONTACT_KEYS
from .commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_PAIR_INDEX = {pair.key: i for i, pair in enumerate(BICYCLE_CONTACT_PAIRS)}
_PAIR = {pair.key: pair for pair in BICYCLE_CONTACT_PAIRS}


class BicyclePSIMotionCommand(MotionCommand):
  """Motion command with a per-phase buffer of physically reached states.

  Candidate capture is staggered across environments, so memory is O(num_envs)
  instead of O(num_envs * validation_horizon).  A captured state is committed
  only after the policy survives the complete future window and satisfies all
  validity gates throughout it.
  """

  cfg: BicyclePSIMotionCommandCfg

  def __init__(self, cfg: BicyclePSIMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if self.bicycle is None:
      raise ValueError("Bicycle PSI requires bicycle_entity_name")
    if cfg.physical_buffer_slots < 1:
      raise ValueError("physical_buffer_slots must be >= 1")
    if cfg.validation_horizon_steps < 1:
      raise ValueError("validation_horizon_steps must be >= 1")
    if not 0.0 <= cfg.reference_sampling_probability <= 1.0:
      raise ValueError("reference_sampling_probability must be in [0, 1]")
    if not 0.0 <= cfg.required_contact_duty <= 1.0:
      raise ValueError("required_contact_duty must be in [0, 1]")
    unknown = set(cfg.required_contact_keys) - set(_PAIR)
    if unknown:
      raise ValueError(f"Unknown PSI required_contact_keys: {sorted(unknown)}")

    # Managers after the command manager (notably actions) do not exist yet.
    # Allocate all state/action-dependent buffers lazily on the first reset.
    self._psi_ready = False
    self.metrics.update(
      {
        "psi_buffer_coverage": torch.zeros(self.num_envs, device=self.device),
        "psi_physical_reset_fraction": torch.zeros(
          self.num_envs, device=self.device
        ),
        "psi_candidate_accept_fraction": torch.zeros(
          self.num_envs, device=self.device
        ),
        "psi_candidate_contact_reject_fraction": torch.zeros(
          self.num_envs, device=self.device
        ),
        "psi_candidate_crank_reject_fraction": torch.zeros(
          self.num_envs, device=self.device
        ),
        "psi_candidate_validity_reject_fraction": torch.zeros(
          self.num_envs, device=self.device
        ),
      }
    )

  def _ensure_psi_buffers(self) -> None:
    if self._psi_ready:
      return
    assert self.bicycle is not None
    nphase = self.motion.time_step_total
    nslot = self.cfg.physical_buffer_slots
    nrj = self.robot.data.joint_pos.shape[1]
    nbj = self.bicycle.data.joint_pos.shape[1]
    na = self._env.action_manager.total_action_dim
    npair = len(self.cfg.required_contact_keys)
    obs_manager = self._env.observation_manager
    self._history_layout = []
    history_size = 0
    for group_name, term_buffers in (
      obs_manager._group_obs_term_history_buffer.items()
    ):
      names = obs_manager._group_obs_term_names[group_name]
      dims = obs_manager._group_obs_term_dim[group_name]
      for term_name, buffer in term_buffers.items():
        term_size = math.prod(dims[names.index(term_name)])
        self._history_layout.append(
          (history_size, history_size + term_size, buffer)
        )
        history_size += term_size

    def phase_slots(dim: int) -> torch.Tensor:
      return torch.zeros(nphase, nslot, dim, device=self.device)

    self._psi_weight = torch.zeros(nphase, nslot, device=self.device)
    self._psi_confirmations = torch.zeros(
      nphase, dtype=torch.long, device=self.device
    )
    self._psi_last_update = torch.zeros(
      nphase, nslot, dtype=torch.long, device=self.device
    )
    self._psi_robot_root = phase_slots(13)
    self._psi_robot_joint_pos = phase_slots(nrj)
    self._psi_robot_joint_vel = phase_slots(nrj)
    self._psi_bike_root = phase_slots(13)
    self._psi_bike_joint_pos = phase_slots(nbj)
    self._psi_bike_joint_vel = phase_slots(nbj)
    self._psi_action = phase_slots(na)
    self._psi_prev_action = phase_slots(na)
    self._psi_prev_prev_action = phase_slots(na)
    self._psi_assistance = phase_slots(1).squeeze(-1)
    self._psi_observation_history = phase_slots(history_size)

    # One pending snapshot per environment.  Capture times are staggered.
    self._pending_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_age = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._pending_phase = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._pending_valid = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._pending_quality = torch.ones(self.num_envs, device=self.device)
    self._pending_schedule_count = torch.zeros(
      self.num_envs, npair, device=self.device
    )
    self._pending_contact_count = torch.zeros_like(self._pending_schedule_count)
    self._pending_pedal_steps = torch.zeros(self.num_envs, device=self.device)
    self._pending_crank = torch.zeros(self.num_envs, device=self.device)
    self._pending_assistance = torch.zeros(self.num_envs, device=self.device)
    self._capture_countdown = torch.randint(
      0,
      self.cfg.validation_horizon_steps,
      (self.num_envs,),
      device=self.device,
    )

    def pending(dim: int) -> torch.Tensor:
      return torch.zeros(self.num_envs, dim, device=self.device)

    self._pending_robot_root = pending(13)
    self._pending_robot_joint_pos = pending(nrj)
    self._pending_robot_joint_vel = pending(nrj)
    self._pending_bike_root = pending(13)
    self._pending_bike_joint_pos = pending(nbj)
    self._pending_bike_joint_vel = pending(nbj)
    self._pending_action = pending(na)
    self._pending_prev_action = pending(na)
    self._pending_prev_prev_action = pending(na)
    self._pending_observation_history = pending(history_size)
    self._candidate_attempts = torch.zeros(self.num_envs, device=self.device)
    self._candidate_accepts = torch.zeros_like(self._candidate_attempts)
    self._candidate_contact_rejects = torch.zeros_like(self._candidate_attempts)
    self._candidate_crank_rejects = torch.zeros_like(self._candidate_attempts)
    self._candidate_validity_rejects = torch.zeros_like(self._candidate_attempts)
    self._psi_global_step = 0
    self._psi_ready = True

  def _observation_histories_initialized(self) -> bool:
    return all(buffer.is_initialized for _, _, buffer in self._history_layout)

  def _capture_observation_history(self, env_ids: torch.Tensor) -> None:
    for start, end, buffer in self._history_layout:
      self._pending_observation_history[env_ids, start:end] = (
        buffer.buffer[env_ids].reshape(env_ids.numel(), -1)
      )

  def _restore_observation_history(
    self, env_ids: torch.Tensor, phases: torch.Tensor, slots: torch.Tensor
  ) -> None:
    """Restore chronological history before the post-reset current append."""
    if env_ids.numel() == 0:
      return
    for start, end, buffer in self._history_layout:
      if buffer._buffer is None:
        # On normal training resets every buffer was initialized by the first
        # observation computation. This guard keeps manual early resets safe.
        continue
      chronological = self._psi_observation_history[
        phases, slots, start:end
      ].reshape(env_ids.numel(), buffer.max_length, *buffer._buffer.shape[2:])
      indices = (
        torch.arange(buffer.max_length, device=self.device)
        + buffer._pointer
        + 1
      ) % buffer.max_length
      internal = torch.empty_like(chronological.transpose(0, 1))
      internal[indices] = chronological.transpose(0, 1)
      buffer._buffer[:, env_ids] = internal
      buffer._num_pushes[env_ids] = buffer.max_length

  def _assistance_scale(self) -> torch.Tensor:
    state = getattr(self._env, "_bicycle_curriculum", None)
    scale = getattr(state, "env_assist_scale", None)
    if scale is None:
      return torch.zeros(self.num_envs, device=self.device)
    if not isinstance(scale, torch.Tensor):
      return torch.full(
        (self.num_envs,), float(scale), device=self.device
      )
    if scale.ndim == 0:
      return scale.expand(self.num_envs)
    return scale

  def _bicycle_state_and_validity(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return validity mask, soft quality, and exact contact-hit channels."""
    assert self.bicycle is not None
    frame_i = self._sim_bike_index("frame")
    frame_q = self.sim_bike_body_quat_w[:, frame_i]
    gravity_w = torch.zeros(self.num_envs, 3, device=self.device)
    gravity_w[:, 2] = -1.0
    gravity_b = quat_apply_inverse(frame_q, gravity_w)
    lean = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2]).abs()
    x_axis = torch.zeros_like(gravity_w)
    x_axis[:, 0] = 1.0
    forward_w = quat_apply(frame_q, x_axis)
    roll_rate = (
      self.bicycle.data.root_link_ang_vel_w * forward_w
    ).sum(dim=-1).abs()
    steer_ids, _ = self.bicycle.find_joints("steer")
    steer = self.bicycle.data.joint_pos[:, steer_ids[0]].abs()

    anchor_error = self.metrics["error_anchor_pos"]
    body_error = self.metrics["error_body_pos"]
    finite = (
      torch.isfinite(self.robot.data.root_link_pose_w).all(dim=-1)
      & torch.isfinite(self.robot.data.root_link_vel_w).all(dim=-1)
      & torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
      & torch.isfinite(self.robot.data.joint_vel).all(dim=-1)
      & torch.isfinite(self.bicycle.data.root_link_pose_w).all(dim=-1)
      & torch.isfinite(self.bicycle.data.root_link_vel_w).all(dim=-1)
      & torch.isfinite(self.bicycle.data.joint_pos).all(dim=-1)
      & torch.isfinite(self.bicycle.data.joint_vel).all(dim=-1)
    )
    valid = (
      finite
      & (anchor_error <= self.cfg.max_anchor_position_error)
      & (body_error <= self.cfg.max_mean_body_position_error)
      & (lean <= self.cfg.max_abs_bicycle_lean)
      & (roll_rate <= self.cfg.max_abs_bicycle_roll_rate)
      & (steer <= self.cfg.max_abs_steering_angle)
      & (
        self._assistance_scale()
        <= self.cfg.max_assistance_scale_for_physical_states
      )
    )

    if self.cfg.forbidden_contact_sensor_name is not None:
      from .bicycle_observations import _contact_force_mag

      forbidden_force = _contact_force_mag(
        self._env, self.cfg.forbidden_contact_sensor_name
      )
      valid &= forbidden_force <= self.cfg.max_forbidden_contact_force

    # A continuous quality in (0.5, 1] is used only for sampling among states
    # that already passed every hard gate.  It is never a reward threshold.
    ratios = torch.stack(
      (
        anchor_error / self.cfg.max_anchor_position_error,
        body_error / self.cfg.max_mean_body_position_error,
        lean / self.cfg.max_abs_bicycle_lean,
        roll_rate / self.cfg.max_abs_bicycle_roll_rate,
        steer / self.cfg.max_abs_steering_angle,
      ),
      dim=-1,
    )
    quality = 0.5 + 0.5 * (1.0 - ratios.amax(dim=-1).clamp(0.0, 1.0))

    from .bicycle_rewards import _contact_pair_signals

    hits = []
    for key in self.cfg.required_contact_keys:
      _, _, _, valid_contact = _contact_pair_signals(
        self._env,
        self,
        key,
        pos_sigma=0.25,
        pos_tolerance=0.02,
        frc_sigma=40.0,
        frc_thres=20.0,
        pedal_z_margin=0.0,
        pedal_z_soft=0.02,
        handle_min_inner_plates=self.cfg.handle_min_inner_plates,
        position_target="simulated",
        pedal_top_contact=False,
        pedal_top_min_local_z=0.0075,
        pedal_sole_contact=self.cfg.pedal_sole_contact,
        force_shaping="tanh",
        force_scale=20.0,
      )
      hits.append(valid_contact)
    contact_hits = (
      torch.stack(hits, dim=-1)
      if hits
      else torch.empty(self.num_envs, 0, device=self.device)
    )
    return valid, quality, contact_hits

  def _capture(self, env_ids: torch.Tensor, crank: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    assert self.bicycle is not None
    origin = self._env.scene.env_origins[env_ids]
    self._pending_active[env_ids] = True
    self._pending_age[env_ids] = 0
    self._pending_phase[env_ids] = self.time_steps[env_ids]
    self._pending_valid[env_ids] = True
    self._pending_quality[env_ids] = 1.0
    self._pending_schedule_count[env_ids] = 0.0
    self._pending_contact_count[env_ids] = 0.0
    self._pending_pedal_steps[env_ids] = 0.0
    self._pending_crank[env_ids] = crank[env_ids]
    self._pending_assistance[env_ids] = self._assistance_scale()[env_ids]

    robot_root = torch.cat(
      (self.robot.data.root_link_pose_w, self.robot.data.root_link_vel_w), dim=-1
    )[env_ids].clone()
    bike_root = torch.cat(
      (self.bicycle.data.root_link_pose_w, self.bicycle.data.root_link_vel_w),
      dim=-1,
    )[env_ids].clone()
    # Store positions relative to the source world so a state can initialize
    # any destination environment without carrying the source env origin.
    robot_root[:, :3] -= origin
    bike_root[:, :3] -= origin
    self._pending_robot_root[env_ids] = robot_root
    self._pending_robot_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
    self._pending_robot_joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
    self._pending_bike_root[env_ids] = bike_root
    self._pending_bike_joint_pos[env_ids] = self.bicycle.data.joint_pos[env_ids]
    self._pending_bike_joint_vel[env_ids] = self.bicycle.data.joint_vel[env_ids]
    actions = self._env.action_manager
    self._pending_action[env_ids] = actions.action[env_ids]
    self._pending_prev_action[env_ids] = actions.prev_action[env_ids]
    self._pending_prev_prev_action[env_ids] = actions.prev_prev_action[env_ids]
    self._capture_observation_history(env_ids)

  def _commit_candidates(
    self,
    env_ids: torch.Tensor,
    contact_ok: torch.Tensor,
    crank_ok: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    phase = self._pending_phase[env_ids]
    valid_ok = self._pending_valid[env_ids]
    accepted = valid_ok & contact_ok & crank_ok
    contact_reject = valid_ok & ~contact_ok
    crank_reject = valid_ok & contact_ok & ~crank_ok
    validity_reject = ~valid_ok
    self._candidate_attempts[env_ids] += 1.0
    self._candidate_accepts[env_ids] += accepted.float()
    self._candidate_contact_rejects[env_ids] += contact_reject.float()
    self._candidate_crank_rejects[env_ids] += crank_reject.float()
    self._candidate_validity_rejects[env_ids] += validity_reject.float()
    denom = self._candidate_attempts[env_ids].clamp_min(1.0)
    self.metrics["psi_candidate_accept_fraction"][env_ids] = (
      self._candidate_accepts[env_ids] / denom
    )
    self.metrics["psi_candidate_contact_reject_fraction"][env_ids] = (
      self._candidate_contact_rejects[env_ids] / denom
    )
    self.metrics["psi_candidate_crank_reject_fraction"][env_ids] = (
      self._candidate_crank_rejects[env_ids] / denom
    )
    self.metrics["psi_candidate_validity_reject_fraction"][env_ids] = (
      self._candidate_validity_rejects[env_ids] / denom
    )

    accepted_ids = env_ids[accepted]
    # Several worlds can propose the same phase. Keep the highest-quality one
    # for this update, then require repeated independent successes per phase.
    if accepted_ids.numel() == 0:
      return
    candidate_phase = self._pending_phase[accepted_ids]
    candidate_quality = self._pending_quality[accepted_ids]
    nphase = self.motion.time_step_total
    best_quality = torch.full((nphase,), -torch.inf, device=self.device)
    best_quality.scatter_reduce_(
      0, candidate_phase, candidate_quality, reduce="amax", include_self=True
    )
    is_best = candidate_quality == best_quality[candidate_phase]
    best_source = torch.full(
      (nphase,), self.num_envs, dtype=torch.long, device=self.device
    )
    best_source.scatter_reduce_(
      0,
      candidate_phase[is_best],
      accepted_ids[is_best],
      reduce="amin",
      include_self=True,
    )
    phases = torch.where(best_source < self.num_envs)[0]
    sources = best_source[phases]
    self._psi_confirmations[phases] += 1
    confirmed = self._psi_confirmations[phases] >= self.cfg.min_confirmations
    slots = torch.argmin(self._effective_weights(phases), dim=-1)
    slots = torch.where(confirmed, slots, torch.zeros_like(slots))

    self._psi_robot_root[phases, slots] = self._pending_robot_root[sources]
    self._psi_robot_joint_pos[phases, slots] = self._pending_robot_joint_pos[sources]
    self._psi_robot_joint_vel[phases, slots] = self._pending_robot_joint_vel[sources]
    self._psi_bike_root[phases, slots] = self._pending_bike_root[sources]
    self._psi_bike_joint_pos[phases, slots] = self._pending_bike_joint_pos[sources]
    self._psi_bike_joint_vel[phases, slots] = self._pending_bike_joint_vel[sources]
    self._psi_action[phases, slots] = self._pending_action[sources]
    self._psi_prev_action[phases, slots] = self._pending_prev_action[sources]
    self._psi_prev_prev_action[phases, slots] = (
      self._pending_prev_prev_action[sources]
    )
    self._psi_assistance[phases, slots] = self._pending_assistance[sources]
    self._psi_observation_history[phases, slots] = (
      self._pending_observation_history[sources]
    )
    confirmed_phases = phases[confirmed]
    confirmed_slots = slots[confirmed]
    confirmed_sources = sources[confirmed]
    self._psi_weight[confirmed_phases, confirmed_slots] = (
      self._pending_quality[confirmed_sources]
    )
    self._psi_last_update[confirmed_phases, confirmed_slots] = self._psi_global_step

  def _effective_weights(self, phase: int | torch.Tensor) -> torch.Tensor:
    age = self._psi_global_step - self._psi_last_update[phase]
    decay = torch.exp(-self.cfg.weight_decay_per_step * age.float())
    return self._psi_weight[phase] * decay

  def _advance_psi(self) -> None:
    if not self.cfg.enabled:
      return
    self._ensure_psi_buffers()
    assert self.bicycle is not None
    self._psi_global_step += 1
    valid, quality, contact_hits = self._bicycle_state_and_validity()
    crank_ids, _ = self.bicycle.find_joints("crank_spin")
    crank = self.bicycle.data.joint_pos[:, crank_ids[0]]

    active = self._pending_active
    if torch.any(active):
      self._pending_age[active] += 1
      self._pending_valid[active] &= valid[active]
      self._pending_quality[active] = torch.minimum(
        self._pending_quality[active], quality[active]
      )
      pair_indices = torch.tensor(
        [_PAIR_INDEX[k] for k in self.cfg.required_contact_keys],
        dtype=torch.long,
        device=self.device,
      )
      schedule = self.ref_contact_mask[:, pair_indices].float()
      self._pending_schedule_count[active] += schedule[active]
      self._pending_contact_count[active] += (
        schedule[active] * contact_hits[active]
      )
      pedal_active = self.ref_contact_mask[:, [
        _PAIR_INDEX["pedal_left"], _PAIR_INDEX["pedal_right"]
      ]].any(dim=-1)
      self._pending_pedal_steps[active] += pedal_active[active].float()

    mature = torch.where(
      self._pending_active
      & (self._pending_age >= self.cfg.validation_horizon_steps)
    )[0]
    if mature.numel() > 0:
      required = self._pending_schedule_count[mature]
      duty = self._pending_contact_count[mature] / required.clamp_min(1.0)
      contact_ok = torch.where(
        required > 0.0,
        duty >= self.cfg.required_contact_duty,
        torch.ones_like(duty, dtype=torch.bool),
      ).all(dim=-1)
      crank_delta = crank[mature] - self._pending_crank[mature]
      crank_ok = (
        (self._pending_pedal_steps[mature] < 1.0)
        | (crank_delta >= self.cfg.min_forward_crank_progress_rad)
      )
      self._commit_candidates(mature, contact_ok, crank_ok)
      self._pending_active[mature] = False
      self._capture_countdown[mature] = torch.randint(
        0,
        self.cfg.validation_horizon_steps,
        (mature.numel(),),
        device=self.device,
      )

    idle = ~self._pending_active
    self._capture_countdown[idle] -= 1
    capture_ids = torch.where(
      idle
      & (self._capture_countdown <= 0)
      & valid
      & self._observation_histories_initialized()
      & (
        self.time_steps
        + self.cfg.validation_horizon_steps
        < self.motion.time_step_total
      )
    )[0]
    self._capture(capture_ids, crank)

    coverage = (self._effective_weights(torch.arange(
      self.motion.time_step_total, device=self.device
    )).amax(dim=-1) > 0.0).float().mean()
    self.metrics["psi_buffer_coverage"][:] = coverage

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self._advance_psi()

  def _discard_pending(self, env_ids: torch.Tensor) -> None:
    if not self._psi_ready:
      return
    self._pending_active[env_ids] = False
    self._pending_age[env_ids] = 0
    self._capture_countdown[env_ids] = torch.randint(
      0,
      self.cfg.validation_horizon_steps,
      (env_ids.numel(),),
      device=self.device,
    )

  def _restore_physical_states(self, env_ids: torch.Tensor) -> None:
    if not self.cfg.enabled or env_ids.numel() == 0:
      return
    self._ensure_psi_buffers()
    assert self.bicycle is not None
    target_assistance = self._assistance_scale()[env_ids]
    assistance_ok = (
      target_assistance
      <= self.cfg.max_assistance_scale_for_physical_states
    )
    phases = self.time_steps[env_ids]
    weights = self._effective_weights(phases)
    assistance_match = (
      self._psi_assistance[phases] - target_assistance.unsqueeze(-1)
    ).abs() <= self.cfg.assistance_match_tolerance
    weights = weights * assistance_match.float()
    available = weights.sum(dim=-1) > 0.0
    choose_physical = (
      available
      & assistance_ok
      & (
        torch.rand(env_ids.numel(), device=self.device)
        >= self.cfg.reference_sampling_probability
      )
    )
    self.metrics["psi_physical_reset_fraction"][env_ids] = (
      choose_physical.float()
    )
    chosen_ids = env_ids[choose_physical]
    if chosen_ids.numel() == 0:
      return
    chosen_phases = phases[choose_physical]
    chosen_slots = torch.multinomial(weights[choose_physical], 1).squeeze(-1)
    origins = self._env.scene.env_origins[chosen_ids]
    robot_root = self._psi_robot_root[chosen_phases, chosen_slots].clone()
    bike_root = self._psi_bike_root[chosen_phases, chosen_slots].clone()
    robot_root[:, :3] += origins
    bike_root[:, :3] += origins
    self.robot.write_root_state_to_sim(robot_root, env_ids=chosen_ids)
    self.robot.write_joint_state_to_sim(
      self._psi_robot_joint_pos[chosen_phases, chosen_slots],
      self._psi_robot_joint_vel[chosen_phases, chosen_slots],
      env_ids=chosen_ids,
    )
    self.bicycle.write_root_state_to_sim(bike_root, env_ids=chosen_ids)
    self.bicycle.write_joint_state_to_sim(
      self._psi_bike_joint_pos[chosen_phases, chosen_slots],
      self._psi_bike_joint_vel[chosen_phases, chosen_slots],
      env_ids=chosen_ids,
    )
    actions = self._env.action_manager
    actions._action[chosen_ids] = self._psi_action[chosen_phases, chosen_slots]
    actions._prev_action[chosen_ids] = self._psi_prev_action[
      chosen_phases, chosen_slots
    ]
    actions._prev_prev_action[chosen_ids] = self._psi_prev_prev_action[
      chosen_phases, chosen_slots
    ]
    self._restore_observation_history(chosen_ids, chosen_phases, chosen_slots)
    self.robot.clear_state(env_ids=chosen_ids)
    self.bicycle.clear_state(env_ids=chosen_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._ensure_psi_buffers()
    self._discard_pending(env_ids)
    super()._resample_command(env_ids)
    self._restore_physical_states(env_ids)

  def reset(self, env_ids: torch.Tensor | None) -> dict[str, float]:
    """Log episode PSI ratios, then clear their private accumulators."""
    assert isinstance(env_ids, torch.Tensor)
    extras = super().reset(env_ids)
    if self._psi_ready:
      self._candidate_attempts[env_ids] = 0.0
      self._candidate_accepts[env_ids] = 0.0
      self._candidate_contact_rejects[env_ids] = 0.0
      self._candidate_crank_rejects[env_ids] = 0.0
      self._candidate_validity_rejects[env_ids] = 0.0
    return extras


@dataclass(kw_only=True)
class BicyclePSIMotionCommandCfg(MotionCommandCfg):
  """Configuration for opt-in, bicycle-aware physical-state initialization."""

  enabled: bool = True
  physical_buffer_slots: int = 2
  reference_sampling_probability: float = 0.34
  validation_horizon_steps: int = 50
  min_confirmations: int = 2
  weight_decay_per_step: float = 1.0e-5

  max_anchor_position_error: float = 0.25
  max_mean_body_position_error: float = 0.25
  max_abs_bicycle_lean: float = 0.45
  max_abs_bicycle_roll_rate: float = 3.0
  max_abs_steering_angle: float = 0.8
  max_assistance_scale_for_physical_states: float = 0.2
  assistance_match_tolerance: float = 0.051

  required_contact_keys: tuple[str, ...] = BICYCLE_SPLIT_CONTACT_KEYS
  required_contact_duty: float = 0.8
  handle_min_inner_plates: int = 2
  pedal_sole_contact: bool = True
  min_forward_crank_progress_rad: float = 0.05

  forbidden_contact_sensor_name: str | None = None
  max_forbidden_contact_force: float = 10.0

  def build(self, env: ManagerBasedRlEnv) -> BicyclePSIMotionCommand:
    return BicyclePSIMotionCommand(self, env)
