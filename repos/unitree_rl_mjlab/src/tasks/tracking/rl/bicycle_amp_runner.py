"""Motion runner that checkpoints the hybrid bicycle AMP discriminator."""

from __future__ import annotations

import torch

from src.tasks.tracking.amp import AmpCfg, get_amp_manager

from .runner import MotionTrackingOnPolicyRunner


class BicycleAmpOnPolicyRunner(MotionTrackingOnPolicyRunner):
  def _amp(self):
    return get_amp_manager(self.env.unwrapped, AmpCfg())

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["bicycle_amp_state"] = self._amp().state_dict()
    torch.save(checkpoint, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    amp_state = checkpoint.get("bicycle_amp_state")
    if amp_state is not None:
      self._amp().load_state_dict(amp_state)
      print(f"[bicycle-amp] restored discriminator state from {path}")
    return infos
