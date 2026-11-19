"""One-way (freewheel) drivetrain coupling for the dynamic bicycle.

The XML ships a *fixed-gear* ``equality`` between ``crank_spin`` and
``rear_spin``. For dynamic bikes we delete that constraint and instead apply a
unilateral soft clutch each step:

  engage only when the crank tries to drive the rear wheel forward
  (``ω_crank > ω_rear / G``), otherwise coast.

This approximates a real freehub: pedaling forward drives the wheel; coasting /
backpedaling does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity

# Chainring:rear ratio in g1_bicycle*.xml (was 2.2; 1.5 matches bike_052 mocap ratchet).
FREEWHEEL_GEAR = 1.5
FREEWHEEL_DAMPING = 8.0  # N·m·s/rad on crank slip when engaged


def apply_bicycle_freewheel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str = "bicycle",
  gear: float = FREEWHEEL_GEAR,
  damping: float = FREEWHEEL_DAMPING,
) -> None:
  """Apply one-way crank→rear coupling torques (step event)."""
  del env_ids
  bicycle: Entity = env.scene[entity_name]
  crank_ids, _ = bicycle.find_joints("crank_spin")
  rear_ids, _ = bicycle.find_joints("rear_spin")
  crank_jid = crank_ids[0]
  rear_jid = rear_ids[0]

  omega_c = bicycle.data.joint_vel[:, crank_jid]
  omega_r = bicycle.data.joint_vel[:, rear_jid]
  # equality was rear = gear * crank  →  equivalent crank rate of wheel = ω_r / gear
  slip = omega_c - omega_r / gear
  engaged = slip > 0.0
  damping_value = getattr(env, "_bicycle_freewheel_damping", damping)
  tau_c = torch.where(engaged, -damping_value * slip, torch.zeros_like(slip))
  tau_r = -tau_c / gear  # power-consistent when locked

  crank_body_ids, _ = bicycle.find_bodies("crank")
  rear_body_ids, _ = bicycle.find_bodies("rear_wheel")
  crank_bid = crank_body_ids[0]
  rear_bid = rear_body_ids[0]

  y_local = torch.zeros(env.num_envs, 3, device=env.device, dtype=tau_c.dtype)
  y_local[:, 1] = 1.0
  axis_c = quat_apply(bicycle.data.body_link_quat_w[:, crank_bid], y_local)
  axis_r = quat_apply(bicycle.data.body_link_quat_w[:, rear_bid], y_local)

  zeros = torch.zeros(env.num_envs, 2, 3, device=env.device, dtype=tau_c.dtype)
  torques = torch.stack(
    [tau_c.unsqueeze(-1) * axis_c, tau_r.unsqueeze(-1) * axis_r],
    dim=1,
  )
  bicycle.write_external_wrench_to_sim(
    zeros,
    torques,
    body_ids=[crank_bid, rear_bid],
  )
