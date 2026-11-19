"""Curriculum domain randomization for the Ratchet V3 bicycle task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .bicycle_curriculum import STEPS_PER_ITER

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


CORE_START_ITER = 25_000.0
CORE_END_ITER = 50_000.0
LATE_START_ITER = 50_000.0
LATE_END_ITER = 75_000.0


def _training_iter(env: ManagerBasedRlEnv) -> float:
  return float(env.common_step_counter) / float(STEPS_PER_ITER)


def _ramp(value: float, start: float, end: float) -> float:
  return min(max((value - start) / max(end - start, 1.0), 0.0), 1.0)


def bicycle_dr_severity(env: ManagerBasedRlEnv) -> tuple[float, float, float]:
  """Return core, inertial (slow core), and late DR severities."""
  it = _training_iter(env)
  core = _ramp(it, CORE_START_ITER, CORE_END_ITER)
  return core, core * core, _ramp(it, LATE_START_ITER, LATE_END_ITER)


def _linear_support(nominal: float, lo: float, hi: float, rho: float) -> tuple[float, float]:
  return nominal + rho * (lo - nominal), nominal + rho * (hi - nominal)


def _mult_support(lo: float, hi: float, rho: float) -> tuple[float, float]:
  """Expand a positive multiplier range around one in log space."""
  return math.exp(rho * math.log(lo)), math.exp(rho * math.log(hi))


def _resolved(env: ManagerBasedRlEnv, cfg: SceneEntityCfg) -> SceneEntityCfg:
  cfg.resolve(env.scene)
  return cfg


@requires_model_fields(
  "geom_friction",
  "dof_damping",
  "dof_frictionloss",
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  "body_subtreemass",
  "dof_invweight0",
  "body_invweight0",
  "tendon_length0",
  "tendon_invweight0",
  "actuator_gainprm",
  "actuator_biasprm",
  # Mass/inertia DR is inactive through iteration 25k.  EventManager's automatic
  # recomputation would nevertheless run at every reset and make the nominal
  # warm-up needlessly expensive, so recompute explicitly only once rho > 0.
  recompute=RecomputeLevel.none,
)
def randomize_bicycle_v3_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  command_name: str = "motion",
) -> None:
  """Resample the current expanding DR support for resetting environments.

  Every model-field operation is relative to mjlab's immutable default model,
  so repeated resets never accumulate randomization.  The reset event runs
  before ``MotionCommand.reset``; updating its ranges here makes the subsequent
  RSI draw use the current curriculum severity.
  """
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if env_ids.numel() == 0:
    return

  core, inertial, late = bicycle_dr_severity(env)
  env._bicycle_dr_core = core
  env._bicycle_dr_inertial = inertial
  env._bicycle_dr_late = late

  # The command is reset after reset-mode events, so this updates the support
  # used by the same reset without independently moving the bicycle.
  command = env.command_manager.get_term(command_name)
  command.cfg.pose_range = {
    "x": (-0.01 * core, 0.01 * core),
    "y": (-0.005 * core, 0.005 * core),
    "z": (0.0, 0.004 * core),
    "roll": (-0.02 * core, 0.02 * core),
    "pitch": (-0.02 * core, 0.02 * core),
    "yaw": (-0.03 * core, 0.03 * core),
  }
  command.cfg.velocity_range = {
    "x": (-0.05 * core, 0.05 * core),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  command.cfg.joint_position_range = (-0.02 * core, 0.02 * core)

  # Preserve the exact Ratchet-V3 dynamics (and reset cost) during the nominal
  # 0--25k lesson.  All three curriculum severities are monotonic, so there is
  # no previously randomized state that needs to be restored here.
  if core <= 0.0 and late <= 0.0:
    return

  # Friction.  Shared draws preserve left/right symmetry within each family.
  friction_specs = (
    ("robot", r"^(left|right)_foot[1-7](_sole)?_collision$", 0.6, 0.35, 1.2, True),
    ("robot", r"^(left|right)_hand_plate_.*_collision$", 1.0, 0.7, 1.6, True),
    ("bicycle", r"^(rear|front)_tire_col$", 1.5, 0.9, 2.0, True),
    ("bicycle", r"^pedal_[rl]$", 1.0, 0.6, 1.6, True),
    ("bicycle", r"^handlebar_col_(left|right)$|^handlebar_thick_col$", 1.0, 0.7, 1.6, True),
    ("bicycle", r"^saddle_[1-5]_col$", 1.0, 0.5, 1.5, True),
    ("terrain", r".*", 1.0, 0.7, 1.4, True),
  )
  for entity, names, nominal, lo, hi, shared in friction_specs:
    bounds = _linear_support(nominal, lo, hi, core)
    dr.geom_friction(
      env,
      env_ids,
      ranges=bounds,
      operation="abs",
      distribution="log_uniform",
      shared_random=shared,
      asset_cfg=_resolved(env, SceneEntityCfg(entity, geom_names=names)),
      axes=[0],
    )

  rolling = _mult_support(0.5, 2.0, core)
  for entity, names in (
    ("bicycle", r"^(rear|front)_tire_col$"),
    ("terrain", r".*"),
  ):
    dr.geom_friction(
      env,
      env_ids,
      ranges=rolling,
      operation="scale",
      distribution="log_uniform",
      shared_random=True,
      asset_cfg=_resolved(env, SceneEntityCfg(entity, geom_names=names)),
      axes=[1, 2],
    )

  # Physics-consistent density scaling. pseudo_inertia uses exp(2 alpha).
  robot_mass = _mult_support(0.95, 1.05, inertial)
  bike_mass = _mult_support(0.92, 1.08, inertial)
  dr.pseudo_inertia(
    env,
    env_ids,
    alpha_range=(0.5 * math.log(robot_mass[0]), 0.5 * math.log(robot_mass[1])),
    asset_cfg=_resolved(env, SceneEntityCfg("robot", body_names=r".*")),
  )
  dr.pseudo_inertia(
    env,
    env_ids,
    alpha_range=(0.5 * math.log(bike_mass[0]), 0.5 * math.log(bike_mass[1])),
    asset_cfg=_resolved(env, SceneEntityCfg("bicycle", body_names=r".*")),
  )

  # Wheel density has the wider 1.7--2.3 kg support around nominal 2.0 kg.
  wheel_mass = _linear_support(1.0, 1.7 / 2.0, 2.3 / 2.0, inertial)
  dr.pseudo_inertia(
    env,
    env_ids,
    alpha_range=(0.5 * math.log(wheel_mass[0]), 0.5 * math.log(wheel_mass[1])),
    asset_cfg=_resolved(
      env, SceneEntityCfg("bicycle", body_names=("rear_wheel", "front_wheel"))
    ),
  )

  com = 0.02 * inertial
  for entity, names in (
    ("robot", ("torso_link",)),
    ("bicycle", ("frame",)),
  ):
    dr.body_com_offset(
      env,
      env_ids,
      ranges={0: (-com, com), 1: (-com, com), 2: (-com, com)},
      operation="add",
      asset_cfg=_resolved(env, SceneEntityCfg(entity, body_names=names)),
    )

  # Bicycle joint dissipation.  All ranges collapse to nominal at core=0.
  joint_specs = (
    ("steer", 3.0, 1.5, 6.0, "damping"),
    ("steer", 0.5, 0.2, 1.0, "friction"),
    ("crank_spin", 0.01, 0.005, 0.03, "damping"),
    (r"pedal_[rl]_spin", 0.02, 0.01, 0.05, "damping"),
  )
  for names, nominal, lo, hi, kind in joint_specs:
    bounds = _linear_support(nominal, lo, hi, inertial)
    func = dr.joint_damping if kind == "damping" else dr.joint_friction
    func(
      env,
      env_ids,
      ranges=bounds,
      operation="abs",
      distribution="log_uniform",
      asset_cfg=_resolved(env, SceneEntityCfg("bicycle", joint_names=names)),
    )

  # The freewheel clutch is implemented in PyTorch rather than a model field.
  freewheel = _linear_support(8.0, 4.0, 14.0, inertial)
  if not hasattr(env, "_bicycle_freewheel_damping"):
    env._bicycle_freewheel_damping = torch.full(
      (env.num_envs,), 8.0, device=env.device
    )
  lo = math.log(max(freewheel[0], 1.0e-6))
  hi = math.log(max(freewheel[1], 1.0e-6))
  env._bicycle_freewheel_damping[env_ids] = torch.exp(
    torch.empty(env_ids.numel(), device=env.device).uniform_(lo, hi)
  )

  dr.encoder_bias(
    env,
    env_ids,
    bias_range=(-0.01 * core, 0.01 * core),
    asset_cfg=_resolved(env, SceneEntityCfg("robot", joint_names=r".*")),
  )

  # Late actuator mismatch. The robot is wrapped in DelayedActuator by the task.
  gains = _mult_support(0.9, 1.1, late)
  dr.pd_gains(
    env,
    env_ids,
    kp_range=gains,
    kd_range=gains,
    distribution="log_uniform",
    operation="scale",
    asset_cfg=SceneEntityCfg("robot"),
  )
  max_lag = int(math.floor(2.0 * late + 1.0e-8))
  dr.sync_actuator_delays(
    env,
    env_ids,
    lag_range=(0, max_lag),
    asset_cfg=SceneEntityCfg("robot"),
  )

  # pseudo_inertia/body_com_offset changed mass properties above.  The model
  # fields are expanded by this function's decorator, but the costly derived
  # constants are rebuilt only after inertial DR has actually started.
  if inertial > 0.0:
    env.sim.recompute_constants(RecomputeLevel.set_const)


def curriculum_observation_noise(
  env: ManagerBasedRlEnv,
  source_func: Callable[..., torch.Tensor],
  source_params: dict[str, Any],
  max_abs: float,
) -> torch.Tensor:
  """Evaluate an observation term and add actor-only core-curriculum noise."""
  value = source_func(env, **source_params)
  core, _, _ = bicycle_dr_severity(env)
  if core <= 0.0 or max_abs <= 0.0:
    return value
  return value + (2.0 * torch.rand_like(value) - 1.0) * (max_abs * core)


def apply_bicycle_dr_push(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  body_name: str = "torso_link",
) -> None:
  """Apply curriculum physical impulses after 50k without setting velocity."""
  del env_ids
  _, _, late = bicycle_dr_severity(env)
  robot = env.scene["robot"]
  body_ids, _ = robot.find_bodies(body_name)
  body_id = body_ids[0]

  if not hasattr(env, "_bicycle_dr_push_left"):
    env._bicycle_dr_push_left = torch.zeros(env.num_envs, device=env.device)
    env._bicycle_dr_push_cooldown = torch.empty(env.num_envs, device=env.device).uniform_(5.0, 10.0)
    env._bicycle_dr_push_force = torch.zeros(env.num_envs, 3, device=env.device)

  left = env._bicycle_dr_push_left
  cooldown = env._bicycle_dr_push_cooldown
  force = env._bicycle_dr_push_force
  dt = float(env.step_dt)
  left.sub_(dt)
  cooldown.sub_(dt)

  expired = left <= 0.0
  force[expired] = 0.0
  ready = expired & (cooldown <= 0.0)
  if ready.any():
    ready_ids = ready.nonzero().flatten()
    trigger = torch.rand(ready_ids.numel(), device=env.device) < late
    trigger_ids = ready_ids[trigger]
    if trigger_ids.numel() > 0 and late > 0.0:
      magnitude = torch.empty(trigger_ids.numel(), device=env.device).uniform_(15.0, 30.0)
      sign = torch.where(
        torch.rand(trigger_ids.numel(), device=env.device) < 0.5,
        -torch.ones_like(magnitude),
        torch.ones_like(magnitude),
      )
      force[trigger_ids, 0] = late * sign * magnitude
      force[trigger_ids, 1] = late * torch.empty_like(magnitude).uniform_(-10.0, 10.0)
      left[trigger_ids] = 0.1
    cooldown[ready_ids] = torch.empty(ready_ids.numel(), device=env.device).uniform_(5.0, 10.0)

  zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
  robot.write_external_wrench_to_sim(
    force.unsqueeze(1), zeros, body_ids=[body_id]
  )
  env._bicycle_dr_push_active = (left > 0.0).float().mean()
  env._bicycle_dr_push_force_abs = torch.linalg.vector_norm(force, dim=-1).mean()


def update_bicycle_dr_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Log DR supports independently from the PhysicsPD curriculum."""
  del env_ids
  core, inertial, late = bicycle_dr_severity(env)
  return {
    "bicycle_dr_core": torch.tensor(core),
    "bicycle_dr_inertial": torch.tensor(inertial),
    "bicycle_dr_late": torch.tensor(late),
    "bicycle_dr_push_active": torch.as_tensor(
      getattr(env, "_bicycle_dr_push_active", 0.0)
    ),
    "bicycle_dr_push_force_abs": torch.as_tensor(
      getattr(env, "_bicycle_dr_push_force_abs", 0.0)
    ),
  }
