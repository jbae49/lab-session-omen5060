"""Runner guard for bicycle DAgger/BC-to-PPO distillation."""

from src.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from .bicycle_dagger_ppo import BicycleDaggerPPO


class BicycleDistillOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Motion runner that requires a frozen teacher before learning."""

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if not isinstance(self.alg, BicycleDaggerPPO):
      raise TypeError("BicycleDistillOnPolicyRunner requires BicycleDaggerPPO")
    if not self.alg.has_teacher:
      raise ValueError(
        "Set --agent.algorithm.teacher-checkpoint to the matching PhysicsPD "
        "teacher .pt file before starting distillation."
      )
    return super().learn(num_learning_iterations, init_at_random_ep_len)

