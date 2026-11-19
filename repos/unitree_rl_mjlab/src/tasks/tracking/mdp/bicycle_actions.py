"""Action terms for kinematic bicycle reference playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BicycleReferenceActionCfg(ActionTermCfg):
  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
    return BicycleReferenceAction(self, env)


class BicycleReferenceAction(ActionTerm):
  """Drive the bicycle entity from the motion command each physics substep."""

  cfg: BicycleReferenceActionCfg

  def __init__(self, cfg: BicycleReferenceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = cast(
      MotionCommand, env.command_manager.get_term(cfg.command_name)
    )

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return torch.zeros(self.num_envs, 0, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions

  def apply_actions(self) -> None:
    self._command.write_bicycle_reference()
