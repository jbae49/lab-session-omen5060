"""VecEnv wrapper: residual_actions + frozen ResMimic base → env.step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from src.tasks.tracking.mdp.resmimic_base import (
  DEFAULT_BASE_POLICY_PATH,
  FrozenResMimicBasePolicy,
)


class ResMimicResidualVecEnvWrapper(RslRlVecEnvWrapper):
  """Policy emits residual; wrapper adds converted ResMimic base actions."""

  def __init__(
    self,
    env,
    *,
    base_policy_path: str | Path | None = None,
    clip_actions: float | None = None,
    command_name: str = "motion",
    residual_scale: float = 1.0,
  ) -> None:
    super().__init__(env, clip_actions=clip_actions)
    path = base_policy_path or DEFAULT_BASE_POLICY_PATH
    # ``env`` here is ManagerBasedRlEnv (or VideoRecorder); unwrap if needed.
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "action_manager"):
      base_env = base_env.env
    self.base = FrozenResMimicBasePolicy(
      path, base_env, command_name=command_name
    )
    self.residual_scale = float(residual_scale)
    print(f"[ResMimicResidual] loaded base policy: {path}")

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    residual = actions
    if self.clip_actions is not None:
      residual = torch.clamp(residual, -self.clip_actions, self.clip_actions)
    residual = residual * self.residual_scale

    base_raw = self.base.act_mjlab_raw(update_history=True)
    total = base_raw + residual
    self.base.adapter.set_last_total_action(total)

    # Parent clips again; pass total through parent step without double residual clip
    # by temporarily disabling clip on the outer call path.
    saved_clip = self.clip_actions
    self.clip_actions = None
    try:
      return super().step(total)
    finally:
      self.clip_actions = saved_clip

  def get_observations(self) -> TensorDict:
    return super().get_observations()
