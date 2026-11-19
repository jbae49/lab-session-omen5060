"""Conditional AMP discriminator for the long middle bicycle-riding phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


MANAGER_ATTR = "_bicycle_hybrid_amp"
EXCLUDED_JOINTS = (
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
KEY_BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_knee_link",
  "right_knee_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


@dataclass
class AmpCfg:
  command_name: str = "motion"
  entity_name: str = "robot"
  command_speed: float = 2.0
  ride_start_frame: int = 608
  ride_end_frame: int = 1608  # half-open
  hidden_dims: tuple[int, ...] = (256, 128)
  grad_penalty: float = 10.0
  learning_rate: float = 1.0e-4
  weight_decay: float = 1.0e-3
  batch_size: int = 256
  buffer_steps: int = 8
  updates_per_step: int = 1


class Discriminator(nn.Module):
  def __init__(
    self, input_dim: int, hidden_dims: tuple[int, ...], grad_penalty: float
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden_dims:
      layers.extend((nn.Linear(width, size), nn.ReLU()))
      width = size
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(width, 1)
    # Start D near zero so Peng's clipped parabola provides a non-zero signal
    # before either side has been learned. A [-1, 1] head saturated this 144-D
    # normalized input at initialization and made the AMP reward exactly zero.
    nn.init.uniform_(self.head.weight, -0.01, 0.01)
    nn.init.zeros_(self.head.bias)
    self.grad_penalty = float(grad_penalty)
    self.register_buffer("feature_mean", torch.zeros(input_dim))
    self.register_buffer("feature_std", torch.ones(input_dim))

  def set_normalizer(self, values: torch.Tensor) -> None:
    self.feature_mean.copy_(values.mean(dim=0))
    self.feature_std.copy_(values.std(dim=0).clamp_min(1.0e-4))

  def _normalize(self, values: torch.Tensor) -> torch.Tensor:
    return (values - self.feature_mean) / self.feature_std

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(self._normalize(values)))

  @torch.no_grad()
  def reward(self, values: torch.Tensor) -> torch.Tensor:
    score = self.forward(values)
    return torch.clamp(1.0 - 0.25 * torch.square(score - 1.0), min=0.0).squeeze(-1)

  def loss(
    self, policy: torch.Tensor, expert: torch.Tensor
  ) -> tuple[torch.Tensor, dict[str, float]]:
    policy_score = self.forward(policy)
    expert_score = self.forward(expert)
    policy_loss = torch.square(policy_score + 1.0).mean()
    expert_loss = torch.square(expert_score - 1.0).mean()
    normalized = self._normalize(expert).detach().requires_grad_(True)
    score = self.head(self.trunk(normalized))
    (gradient,) = torch.autograd.grad(
      score,
      normalized,
      torch.ones_like(score),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )
    penalty = self.grad_penalty * gradient.square().sum(dim=-1).mean()
    total = policy_loss + expert_loss + penalty
    return total, {
      "policy_loss": float(policy_loss.detach()),
      "expert_loss": float(expert_loss.detach()),
      "grad_penalty": float(penalty.detach()),
      "policy_score": float(policy_score.mean().detach()),
      "expert_score": float(expert_score.mean().detach()),
    }


@dataclass
class AmpLayout:
  joint_ids: torch.Tensor
  pelvis_id: int
  key_body_ids: torch.Tensor
  obs_dim: int

  @classmethod
  def build(cls, env: ManagerBasedRlEnv, entity_name: str) -> "AmpLayout":
    robot = env.scene[entity_name]
    joint_names = [name for name in robot.joint_names if name not in EXCLUDED_JOINTS]
    joint_ids = torch.tensor(
      [robot.joint_names.index(name) for name in joint_names],
      device=env.device,
      dtype=torch.long,
    )
    key_ids = torch.tensor(
      [robot.body_names.index(name) for name in KEY_BODIES],
      device=env.device,
      dtype=torch.long,
    )
    return cls(
      joint_ids=joint_ids,
      pelvis_id=robot.body_names.index("pelvis"),
      key_body_ids=key_ids,
      obs_dim=2 * len(joint_names) + 3 * len(KEY_BODIES) + 3 + 1,
    )


def _sim_features(
  env: ManagerBasedRlEnv, layout: AmpLayout, entity_name: str, speed: float
) -> torch.Tensor:
  robot = env.scene[entity_name]
  data = robot.data
  pelvis_pos = data.body_link_pos_w[:, layout.pelvis_id]
  pelvis_quat = data.body_link_quat_w[:, layout.pelvis_id]
  relative = data.body_link_pos_w[:, layout.key_body_ids] - pelvis_pos[:, None]
  local = quat_apply_inverse(
    pelvis_quat[:, None].expand(-1, len(KEY_BODIES), -1), relative
  )
  gravity = torch.zeros(env.num_envs, 3, device=env.device)
  gravity[:, 2] = -1.0
  projected = quat_apply_inverse(pelvis_quat, gravity)
  command = torch.full((env.num_envs, 1), float(speed), device=env.device)
  return torch.cat(
    (
      data.joint_pos[:, layout.joint_ids],
      data.joint_vel[:, layout.joint_ids],
      local.flatten(1),
      projected,
      command,
    ),
    dim=-1,
  )


def _demo_features(
  env: ManagerBasedRlEnv, cfg: AmpCfg, layout: AmpLayout
) -> torch.Tensor:
  command = env.command_manager.get_term(cfg.command_name)
  motion = command.motion
  frames = slice(cfg.ride_start_frame, cfg.ride_end_frame)
  joint_pos = motion.joint_pos[frames, layout.joint_ids]
  joint_vel = motion.joint_vel[frames, layout.joint_ids]
  pelvis_pos = motion._body_pos_w[frames, layout.pelvis_id]
  pelvis_quat = motion._body_quat_w[frames, layout.pelvis_id]
  relative = motion._body_pos_w[frames][:, layout.key_body_ids] - pelvis_pos[:, None]
  local = quat_apply_inverse(
    pelvis_quat[:, None].expand(-1, len(KEY_BODIES), -1), relative
  )
  gravity = torch.zeros(pelvis_quat.shape[0], 3, device=env.device)
  gravity[:, 2] = -1.0
  projected = quat_apply_inverse(pelvis_quat, gravity)
  speed = torch.full(
    (pelvis_quat.shape[0], 1), float(cfg.command_speed), device=env.device
  )
  return torch.cat(
    (joint_pos, joint_vel, local.flatten(1), projected, speed), dim=-1
  )


@dataclass
class BicycleAmpManager:
  cfg: AmpCfg
  layout: AmpLayout
  expert_pairs: torch.Tensor
  discriminator: Discriminator
  optimizer: torch.optim.Optimizer
  previous: torch.Tensor
  previous_riding: torch.Tensor
  buffer: torch.Tensor
  buffer_fill: int = 0
  buffer_head: int = 0
  info: dict[str, float] = field(default_factory=dict)

  @classmethod
  def build(cls, env: ManagerBasedRlEnv, cfg: AmpCfg) -> "BicycleAmpManager":
    with torch.inference_mode(False), torch.enable_grad():
      layout = AmpLayout.build(env, cfg.entity_name)
      demo = _demo_features(env, cfg, layout)
      expert_pairs = torch.cat((demo[:-1], demo[1:]), dim=-1).clone()
      discriminator = Discriminator(
        2 * layout.obs_dim, cfg.hidden_dims, cfg.grad_penalty
      ).to(env.device)
      discriminator.set_normalizer(expert_pairs)
      optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
      )
      capacity = cfg.buffer_steps * env.num_envs
      print(
        f"[bicycle-amp] {expert_pairs.shape[0]} expert transitions, "
        f"{2 * layout.obs_dim}D, ride=[{cfg.ride_start_frame},{cfg.ride_end_frame})"
      )
      return cls(
        cfg=cfg,
        layout=layout,
        expert_pairs=expert_pairs,
        discriminator=discriminator,
        optimizer=optimizer,
        previous=torch.zeros(env.num_envs, layout.obs_dim, device=env.device),
        previous_riding=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        buffer=torch.zeros(capacity, 2 * layout.obs_dim, device=env.device),
      )

  def style_reward(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    current = _sim_features(
      env, self.layout, self.cfg.entity_name, self.cfg.command_speed
    )
    command = env.command_manager.get_term(self.cfg.command_name)
    riding = (command.time_steps >= self.cfg.ride_start_frame) & (
      command.time_steps < self.cfg.ride_end_frame
    )
    fresh = env.episode_length_buf <= 1
    valid = riding & self.previous_riding & ~fresh
    pairs = torch.cat((self.previous, current), dim=-1)
    reward = self.discriminator.reward(pairs) * valid.float()
    if valid.any():
      self._push(pairs[valid])
      for _ in range(self.cfg.updates_per_step):
        self._update()
    self.previous.copy_(current)
    self.previous_riding.copy_(riding)
    log = env.extras.get("log")
    if log is not None:
      log.update(self.info)
      log["amp/style_reward"] = float(reward[valid].mean()) if valid.any() else 0.0
      log["amp/riding_fraction"] = float(riding.float().mean())
    return reward

  def _push(self, pairs: torch.Tensor) -> None:
    count = min(pairs.shape[0], self.buffer.shape[0])
    end = self.buffer_head + count
    if end <= self.buffer.shape[0]:
      self.buffer[self.buffer_head:end] = pairs[:count]
    else:
      split = self.buffer.shape[0] - self.buffer_head
      self.buffer[self.buffer_head:] = pairs[:split]
      self.buffer[: count - split] = pairs[split:count]
    self.buffer_head = end % self.buffer.shape[0]
    self.buffer_fill = min(self.buffer_fill + count, self.buffer.shape[0])

  def _update(self) -> None:
    if self.buffer_fill < self.cfg.batch_size:
      return
    with torch.inference_mode(False), torch.enable_grad():
      policy_ids = torch.randint(
        self.buffer_fill, (self.cfg.batch_size,), device=self.buffer.device
      )
      expert_ids = torch.randint(
        self.expert_pairs.shape[0],
        (self.cfg.batch_size,),
        device=self.buffer.device,
      )
      policy = self.buffer[policy_ids].clone()
      expert = self.expert_pairs[expert_ids].clone()
      loss, info = self.discriminator.loss(policy, expert)
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
      self.optimizer.step()
      self.info = {f"amp/{key}": value for key, value in info.items()}
      self.info["amp/loss"] = float(loss.detach())

  def state_dict(self) -> dict:
    return {
      "discriminator": self.discriminator.state_dict(),
      "optimizer": self.optimizer.state_dict(),
      "buffer": self.buffer[: self.buffer_fill].detach().cpu(),
      "buffer_fill": self.buffer_fill,
      "buffer_head": self.buffer_head,
    }

  def load_state_dict(self, state: dict) -> None:
    self.discriminator.load_state_dict(state["discriminator"], strict=True)
    self.optimizer.load_state_dict(state["optimizer"])
    saved = state.get("buffer")
    if saved is not None:
      count = min(saved.shape[0], self.buffer.shape[0])
      self.buffer[:count] = saved[:count].to(self.buffer.device)
      self.buffer_fill = count
      self.buffer_head = count % self.buffer.shape[0]


def get_amp_manager(
  env: ManagerBasedRlEnv, cfg: AmpCfg | None = None
) -> BicycleAmpManager:
  manager = getattr(env, MANAGER_ATTR, None)
  if manager is None:
    manager = BicycleAmpManager.build(env, cfg or AmpCfg())
    setattr(env, MANAGER_ATTR, manager)
  return manager
