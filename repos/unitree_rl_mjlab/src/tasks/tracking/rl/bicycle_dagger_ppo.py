"""Frozen privileged-teacher DAgger/BC to PPO for bicycle tracking."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


_TEACHER_MEAN_KEY = "_distill_teacher_mean"
_TEACHER_STD_KEY = "_distill_teacher_std"
_STUDENT_ACTION_MASK_KEY = "_distill_student_action_mask"


def _actor_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
  """Read current or legacy mjlab actor weights without loading critic/optimizer."""
  if "actor_state_dict" in checkpoint:
    state = dict(checkpoint["actor_state_dict"])
  elif "model_state_dict" in checkpoint:
    state = {}
    for key, value in checkpoint["model_state_dict"].items():
      if key.startswith("actor."):
        state[key.replace("actor.", "mlp.", 1)] = value
      elif key.startswith("actor_obs_normalizer."):
        state[key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = value
      elif key in ("std", "log_std"):
        state[key] = value
  else:
    raise KeyError("Teacher checkpoint has neither actor_state_dict nor model_state_dict")

  # rsl_rl 4.x -> 5.x Gaussian distribution parameter names.
  if "std" in state:
    state["distribution.std_param"] = state.pop("std")
  if "log_std" in state:
    state["distribution.log_std_param"] = state.pop("log_std")
  return state


class BicycleDaggerPPO(PPO):
  """PPO with a frozen privileged MLP teacher and scheduled BC/DAgger.

  DAgger uses a per-environment Bernoulli choice, not an interpolated action.
  The rollout stores the student's sampled action/log-probability and masks PPO
  actor loss on teacher-executed rows. This avoids treating a teacher action as
  an on-policy student sample while still training the asymmetric critic on all
  collected rewards.
  """

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *,
    teacher: MLPModel | None,
    teacher_checkpoint: str = "",
    beta_hold_iterations: int = 100,
    beta_decay_iterations: int = 1000,
    bc_only_iterations: int = 1000,
    critic_warmup_iterations: int = 200,
    ppo_ramp_iterations: int = 800,
    bc_final_coef: float = 0.1,
    bc_std_loss_coef: float = 1.0,
    **ppo_kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **ppo_kwargs)
    if min(
      beta_hold_iterations,
      beta_decay_iterations,
      bc_only_iterations,
      critic_warmup_iterations,
      ppo_ramp_iterations,
    ) < 0:
      raise ValueError("Distillation schedule iteration counts must be non-negative")
    if not 0.0 <= bc_final_coef <= 1.0:
      raise ValueError("bc_final_coef must be in [0, 1]")

    self.teacher = teacher
    self.teacher_checkpoint = teacher_checkpoint
    self.beta_hold_iterations = beta_hold_iterations
    self.beta_decay_iterations = beta_decay_iterations
    self.bc_only_iterations = bc_only_iterations
    self.critic_warmup_iterations = critic_warmup_iterations
    self.ppo_ramp_iterations = ppo_ramp_iterations
    self.bc_final_coef = bc_final_coef
    self.bc_std_loss_coef = bc_std_loss_coef
    self.distill_iteration = 0
    self._last_teacher_fraction = 0.0

  @property
  def has_teacher(self) -> bool:
    return self.teacher is not None

  def _schedule(self) -> tuple[float, float, float, float]:
    """Return beta, BC coefficient, PPO actor coefficient, critic coefficient."""
    it = self.distill_iteration
    if self.beta_decay_iterations == 0:
      beta = float(it < self.beta_hold_iterations)
    else:
      beta = 1.0 - max(it - self.beta_hold_iterations, 0) / self.beta_decay_iterations
      beta = min(max(beta, 0.0), 1.0)

    if it < self.bc_only_iterations:
      return beta, 1.0, 0.0, 0.0

    warmup_progress = (
      1.0
      if self.critic_warmup_iterations == 0
      else min((it - self.bc_only_iterations) / self.critic_warmup_iterations, 1.0)
    )
    ppo_start = self.bc_only_iterations + self.critic_warmup_iterations
    ppo_progress = (
      1.0
      if self.ppo_ramp_iterations == 0
      else min(max(it - ppo_start, 0) / self.ppo_ramp_iterations, 1.0)
    )
    bc_coef = max(1.0 - ppo_progress, self.bc_final_coef)
    return beta, bc_coef, ppo_progress, warmup_progress

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Query both policies, record student data, and execute Bernoulli-mixed actions."""
    self.transition.hidden_states = (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    student_action = self.actor(obs, stochastic_output=True).detach()
    self.transition.actions = student_action
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(student_action).detach()
    self.transition.distribution_params = tuple(
      p.detach() for p in self.actor.output_distribution_params
    )

    beta, _, _, _ = self._schedule()
    if self.teacher is None:
      teacher_mean = self.actor.output_mean.detach()
      teacher_std = self.actor.output_std.detach().expand_as(teacher_mean)
      student_mask = torch.ones(
        student_action.shape[0], 1, device=student_action.device
      )
      actions_to_env = student_action
    else:
      # stochastic_output=True initializes the frozen teacher distribution;
      # its sample is discarded because DAgger executes deterministic teacher mu.
      self.teacher(obs, stochastic_output=True)
      teacher_mean = self.teacher.output_mean.detach()
      teacher_std = self.teacher.output_std.detach().expand_as(teacher_mean)
      teacher_mask = torch.rand(
        student_action.shape[0], 1, device=student_action.device
      ) < beta
      student_mask = (~teacher_mask).float()
      actions_to_env = torch.where(teacher_mask, teacher_mean, student_action)
      self._last_teacher_fraction = teacher_mask.float().mean().item()

    # These auxiliary tensors are stored by RolloutStorage but are not members
    # of any model obs set, so they cannot leak privileged teacher outputs into
    # the student actor or critic.
    obs[_TEACHER_MEAN_KEY] = teacher_mean
    obs[_TEACHER_STD_KEY] = teacher_std
    obs[_STUDENT_ACTION_MASK_KEY] = student_mask
    self.transition.observations = obs
    return actions_to_env

  @staticmethod
  def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim > value.ndim:
      mask = mask.squeeze(-1)
    while mask.ndim < value.ndim:
      mask = mask.unsqueeze(-1)
    denom = mask.expand_as(value).sum().clamp_min(1.0)
    return (value * mask).sum() / denom

  def update(self) -> dict[str, float]:
    """Optimize BC, critic, and PPO actor using the current fixed schedule."""
    if self.teacher is None:
      raise RuntimeError(
        "Distillation training requires --agent.algorithm.teacher-checkpoint"
      )
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise NotImplementedError("The first bicycle student is intentionally MLP-only")
    if self.rnd or self.symmetry:
      raise NotImplementedError("RND/symmetry are not supported by BicycleDaggerPPO")

    beta, bc_coef, ppo_coef, critic_coef = self._schedule()
    totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "bc_mean": 0.0, "bc_std": 0.0}
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1e-8)

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      new_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      student_mask = batch.observations[_STUDENT_ACTION_MASK_KEY]
      if ppo_coef > 0.0 and self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, new_params)
          kl_mean = self._masked_mean(kl, student_mask)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = self._masked_mean(
        torch.maximum(surrogate, surrogate_clipped), student_mask.squeeze(-1)
      )

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_loss = torch.maximum(
          (values - batch.returns).pow(2),
          (value_clipped - batch.returns).pow(2),
        ).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      student_mean = self.actor.output_mean
      student_std = self.actor.output_std.expand_as(student_mean)
      teacher_mean = batch.observations[_TEACHER_MEAN_KEY]
      teacher_std = batch.observations[_TEACHER_STD_KEY]
      bc_mean_loss = (student_mean - teacher_mean).pow(2).mean()
      bc_std_loss = (student_std - teacher_std).pow(2).mean()
      bc_loss = bc_mean_loss + self.bc_std_loss_coef * bc_std_loss

      loss = (
        bc_coef * bc_loss
        + ppo_coef * (surrogate_loss - self.entropy_coef * entropy.mean())
        + critic_coef * self.value_loss_coef * value_loss
      )
      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      totals["value"] += value_loss.item()
      totals["surrogate"] += surrogate_loss.item()
      totals["entropy"] += entropy.mean().item()
      totals["bc_mean"] += bc_mean_loss.item()
      totals["bc_std"] += bc_std_loss.item()

    updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self.distill_iteration += 1
    result = {name: value / updates for name, value in totals.items()}
    result.update(
      {
        "distill_beta": beta,
        "distill_teacher_fraction": self._last_teacher_fraction,
        "distill_bc_coef": bc_coef,
        "distill_ppo_coef": ppo_coef,
        "distill_critic_coef": critic_coef,
      }
    )
    return result

  def train_mode(self) -> None:
    super().train_mode()
    if self.teacher is not None:
      self.teacher.eval()

  def save(self) -> dict:
    state = super().save()
    state["distill_iteration"] = self.distill_iteration
    state["teacher_checkpoint"] = self.teacher_checkpoint
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    self.distill_iteration = int(loaded_dict.get("distill_iteration", 0))
    return load_iteration

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> "BicycleDaggerPPO":
    """Construct student/critic plus a same-architecture frozen teacher."""
    alg_cfg = cfg["algorithm"]
    alg_cfg.pop("class_name", None)
    alg_cfg.pop("share_cnn_encoders", None)
    # Standard PPO.construct_algorithm injects these resolved optional keys.
    # This custom constructor intentionally does not support either feature,
    # but rsl_rl's runner/logger still expects the keys to exist.
    alg_cfg.setdefault("rnd_cfg", None)
    alg_cfg.setdefault("symmetry_cfg", None)
    actor_cfg = copy.deepcopy(cfg["actor"])
    critic_cfg = copy.deepcopy(cfg["critic"])
    for model_cfg in (actor_cfg, critic_cfg):
      for optional_key in ("cnn_cfg", "distribution_cfg"):
        if model_cfg.get(optional_key) is None:
          model_cfg.pop(optional_key, None)
    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
    critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore

    cfg["obs_groups"] = resolve_obs_groups(
      obs, cfg["obs_groups"], ["actor", "critic", "teacher"]
    )
    student = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **copy.deepcopy(actor_cfg)
    ).to(device)
    critic = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_cfg
    ).to(device)

    teacher_checkpoint = str(alg_cfg.get("teacher_checkpoint", ""))
    teacher: MLPModel | None = None
    if teacher_checkpoint:
      checkpoint_path = Path(teacher_checkpoint).expanduser().resolve()
      if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
      teacher = actor_class(
        obs,
        cfg["obs_groups"],
        "teacher",
        env.num_actions,
        **copy.deepcopy(actor_cfg),
      ).to(device)
      checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
      try:
        teacher.load_state_dict(_actor_state_dict(checkpoint), strict=True)
      except RuntimeError as exc:
        raise RuntimeError(
          "Teacher checkpoint is incompatible with the selected distill task. "
          "Use the student task whose V1/V2/V3/V4 variant matches the teacher."
        ) from exc
      teacher.eval()
      teacher.requires_grad_(False)
      alg_cfg["teacher_checkpoint"] = str(checkpoint_path)
      print(f"Frozen teacher loaded: {checkpoint_path}")

    # Auxiliary rollout targets live in the TensorDict storage but no network's
    # obs_groups references them.
    storage_obs = obs.clone()
    storage_obs[_TEACHER_MEAN_KEY] = torch.zeros(
      env.num_envs, env.num_actions, device=device
    )
    storage_obs[_TEACHER_STD_KEY] = torch.zeros(
      env.num_envs, env.num_actions, device=device
    )
    storage_obs[_STUDENT_ACTION_MASK_KEY] = torch.ones(
      env.num_envs, 1, device=device
    )
    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      storage_obs,
      [env.num_actions],
      device,
    )
    algorithm = BicycleDaggerPPO(
      student,
      critic,
      storage,
      teacher=teacher,
      device=device,
      multi_gpu_cfg=cfg.get("multi_gpu"),
      **alg_cfg,
    )
    print(f"Student Actor Model: {student}")
    print(f"Asymmetric Critic Model: {critic}")
    if teacher is not None:
      print(f"Privileged Teacher Model (frozen): {teacher}")
    return algorithm
