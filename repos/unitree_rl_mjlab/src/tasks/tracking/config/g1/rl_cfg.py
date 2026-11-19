"""RL configuration for Unitree G1 tracking task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
  )


def unitree_g1_bicycle_physics_pd_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Long-horizon runner for PD decay followed by zero-assistance training."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.max_iterations = 100_000
  return cfg


def unitree_g1_bicycle_50k_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """50k runner for deployable-observation bicycle ablations."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.max_iterations = 50_000
  return cfg


@dataclass
class BicycleDualLatentModelCfg(RslRlModelCfg):
  """Configuration fields consumed by the dual privileged actor."""

  class_name: str = (
    "src.tasks.tracking.rl.bicycle_dual_latent_model:"
    "BicycleDualPrivilegedEncoderModel"
  )
  encoder_hidden_dims: tuple[int, ...] = (256, 128)
  latent_dim: int = 8


def unitree_g1_bicycle_dual_latent_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Ratchet V3 PPO with independent 256-128-8 robot/bicycle encoders."""
  cfg = unitree_g1_bicycle_physics_pd_ppo_runner_cfg()
  cfg.actor = BicycleDualLatentModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    encoder_hidden_dims=(256, 128),
    latent_dim=8,
  )
  cfg.obs_groups = {
    "actor": ("actor", "robot_privileged", "bicycle_privileged"),
    "critic": ("critic",),
  }
  return cfg


@dataclass
class BicycleDaggerPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Frozen-teacher DAgger/BC-to-PPO schedule (mjlab PPO iterations)."""

  class_name: str = (
    "src.tasks.tracking.rl.bicycle_dagger_ppo:BicycleDaggerPPO"
  )
  teacher_checkpoint: str = ""
  beta_hold_iterations: int = 100
  beta_decay_iterations: int = 1000
  bc_only_iterations: int = 1000
  critic_warmup_iterations: int = 200
  ppo_ramp_iterations: int = 800
  bc_final_coef: float = 0.1
  bc_std_loss_coef: float = 1.0


def unitree_g1_bicycle_distill_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Runner config for one-run DAgger/BC -> asymmetric PPO distillation."""
  base = unitree_g1_tracking_ppo_runner_cfg()
  base.algorithm = BicycleDaggerPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )
  base.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic",),
    "teacher": ("teacher",),
  }
  # Keep the teacher task's 50k PhysicsPD decay meaningful, then leave another
  # 50k iterations for fully unassisted student PPO, just like the teacher run.
  base.max_iterations = 100_000
  base.save_interval = 500
  return base
