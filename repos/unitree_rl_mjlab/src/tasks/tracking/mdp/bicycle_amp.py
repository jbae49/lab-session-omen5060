"""Phase gates and reward hooks for hybrid DeepMimic/AMP bicycle learning."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch

from src.tasks.tracking.amp import AmpCfg, get_amp_manager

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _ride_weight(
  frame: torch.Tensor, start: int, end: int, blend_frames: int
) -> torch.Tensor:
  weight = ((frame >= start) & (frame < end)).float()
  if blend_frames <= 0:
    return weight
  entering = (frame >= start) & (frame < start + blend_frames)
  u_in = (frame - start).float() / float(blend_frames)
  weight = torch.where(
    entering, 0.5 * (1.0 - torch.cos(torch.pi * u_in)), weight
  )
  leaving = (frame >= end - blend_frames) & (frame < end)
  u_out = (end - frame).float() / float(blend_frames)
  weight = torch.where(
    leaving, 0.5 * (1.0 - torch.cos(torch.pi * u_out)), weight
  )
  return weight


def hybrid_phase_scaled_reward(
  env: ManagerBasedRlEnv,
  source_func: Callable[..., torch.Tensor],
  source_params: dict[str, Any],
  command_name: str = "motion",
  ride_start_frame: int = 608,
  ride_end_frame: int = 1608,
  ride_scale: float = 0.2,
  blend_frames: int = 10,
) -> torch.Tensor:
  """Keep DeepMimic tracking at a weak anchor during the AMP ride."""
  reward = source_func(env, **source_params)
  command = env.command_manager.get_term(command_name)
  ride = _ride_weight(
    command.time_steps, ride_start_frame, ride_end_frame, blend_frames
  )
  scale = 1.0 - ride * (1.0 - float(ride_scale))
  return reward * scale


def bicycle_amp_style_reward(
  env: ManagerBasedRlEnv, cfg: AmpCfg | None = None
) -> torch.Tensor:
  return get_amp_manager(env, cfg).style_reward(env)


def bicycle_fixed_speed_command(
  env: ManagerBasedRlEnv, speed: float = 2.0
) -> torch.Tensor:
  return torch.full((env.num_envs, 1), float(speed), device=env.device)
