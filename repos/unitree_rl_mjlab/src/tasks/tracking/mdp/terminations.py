from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .bicycle_data import (
  BICYCLE_CONTACT_PAIRS,
  BICYCLE_CONTACT_PAIR_KEYS,
  FOOT_SOLE_CONTACT_LOCAL,
)
from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


class BicyclePedalContactMissingTermination:
  """Terminate after either scheduled foot-pedal contact is absent N steps."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._missing_steps = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._missing_steps[env_ids] = 0

  @staticmethod
  def _has_contact(sensor: ContactSensor, force_threshold: float) -> torch.Tensor:
    found = sensor.data.found
    if found is None:
      raise RuntimeError(f"Sensor {sensor.cfg.name!r} must provide found")
    current = found.reshape(found.shape[0], -1).any(dim=-1)
    history = sensor.data.force_history
    if history is None:
      return current
    # Count a contact seen at any physics substep in the current 50 Hz policy
    # interval so a substep-phase artifact cannot increment the miss streak.
    recent = (torch.norm(history, dim=-1) > float(force_threshold)).any(dim=(1, 2))
    return current | recent

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    left_sensor_name: str,
    right_sensor_name: str,
    max_missing_steps: int = 10,
    force_threshold: float = 0.1,
  ) -> torch.Tensor:
    if max_missing_steps <= 0:
      raise ValueError("max_missing_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    pair_indices = (
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_left"),
      BICYCLE_CONTACT_PAIR_KEYS.index("pedal_right"),
    )
    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    contacts = torch.stack(
      (
        self._has_contact(env.scene[left_sensor_name], force_threshold),
        self._has_contact(env.scene[right_sensor_name], force_threshold),
      ),
      dim=-1,
    )
    missing = scheduled & ~contacts
    self._missing_steps = torch.where(
      missing, self._missing_steps + 1, torch.zeros_like(self._missing_steps)
    )
    return (self._missing_steps >= int(max_missing_steps)).any(dim=-1)


class BicyclePedalDistanceMissingTermination:
  """Terminate when either scheduled sole point stays far from its pedal.

  This deliberately ignores contact sensors.  Distance is measured between
  the same mid-sole point used by the pedal contact reward and the designated
  interaction point on the *simulated*, rotating left/right pedal.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._far_steps = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._far_steps[env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    max_distance: float = 0.10,
    max_far_steps: int = 10,
  ) -> torch.Tensor:
    if max_distance <= 0.0:
      raise ValueError("max_distance must be positive")
    if max_far_steps <= 0:
      raise ValueError("max_far_steps must be positive")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if command.bicycle is None:
      raise RuntimeError(f"{command_name!r} has no bicycle entity")

    distances: list[torch.Tensor] = []
    pair_indices: list[int] = []
    sole_local = torch.tensor(
      FOOT_SOLE_CONTACT_LOCAL,
      device=env.device,
      dtype=command.robot_body_pos_w.dtype,
    )
    for key in ("pedal_left", "pedal_right"):
      pair_idx = BICYCLE_CONTACT_PAIR_KEYS.index(key)
      pair = BICYCLE_CONTACT_PAIRS[pair_idx]
      pair_indices.append(pair_idx)

      foot_pos = command.robot_body_pos_by_name(pair.ee_body)
      foot_quat = command.robot_body_quat_by_name(pair.ee_body)
      sole_point = foot_pos + quat_apply(
        foot_quat, sole_local.expand(foot_pos.shape[0], -1)
      )

      bike_idx = command._sim_bike_index(pair.bike_body)
      pedal_pos = command.sim_bike_body_pos_w[:, bike_idx]
      pedal_quat = command.sim_bike_body_quat_w[:, bike_idx]
      if pair.target_local_pos is not None:
        pedal_local = torch.tensor(
          pair.target_local_pos,
          device=env.device,
          dtype=pedal_pos.dtype,
        )
        pedal_pos = pedal_pos + quat_apply(
          pedal_quat, pedal_local.expand(pedal_pos.shape[0], -1)
        )
      distances.append(torch.norm(sole_point - pedal_pos, dim=-1))

    scheduled = command.ref_contact_mask[:, pair_indices].bool()
    far = scheduled & (torch.stack(distances, dim=-1) > float(max_distance))
    self._far_steps = torch.where(
      far, self._far_steps + 1, torch.zeros_like(self._far_steps)
    )
    return (self._far_steps >= int(max_far_steps)).any(dim=-1)


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)
