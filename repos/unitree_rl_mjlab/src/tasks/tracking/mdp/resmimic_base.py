"""Frozen ResMimic base policy: obs adapter + action-scale conversion.

ResMimic student_future base expects 1757-D obs:
  current(127) + history×10(1270) + future(360)
and outputs 29-D actions with ``q = q_default_resmimic + 0.5 * a``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# From ResMimic g1_mimic_distill_config.default_joint_angles
RESMIMIC_DEFAULT_JOINT_POS: dict[str, float] = {
  "left_hip_pitch_joint": -0.2,
  "left_hip_roll_joint": 0.0,
  "left_hip_yaw_joint": 0.0,
  "left_knee_joint": 0.4,
  "left_ankle_pitch_joint": -0.2,
  "left_ankle_roll_joint": 0.0,
  "right_hip_pitch_joint": -0.2,
  "right_hip_roll_joint": 0.0,
  "right_hip_yaw_joint": 0.0,
  "right_knee_joint": 0.4,
  "right_ankle_pitch_joint": -0.2,
  "right_ankle_roll_joint": 0.0,
  "waist_yaw_joint": 0.0,
  "waist_roll_joint": 0.0,
  "waist_pitch_joint": 0.0,
  "left_shoulder_pitch_joint": 0.0,
  "left_shoulder_roll_joint": 0.4,
  "left_shoulder_yaw_joint": 0.0,
  "left_elbow_joint": 1.2,
  "left_wrist_roll_joint": 0.0,
  "left_wrist_pitch_joint": 0.0,
  "left_wrist_yaw_joint": 0.0,
  "right_shoulder_pitch_joint": 0.0,
  "right_shoulder_roll_joint": -0.4,
  "right_shoulder_yaw_joint": 0.0,
  "right_elbow_joint": 1.2,
  "right_wrist_roll_joint": 0.0,
  "right_wrist_pitch_joint": 0.0,
  "right_wrist_yaw_joint": 0.0,
}

RESMIMIC_ACTION_SCALE = 0.5
RESMIMIC_N_OBS_SINGLE = 127  # mimic35 + proprio92
RESMIMIC_HISTORY_LEN = 10
RESMIMIC_FUTURE_STEPS: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
RESMIMIC_OBS_DIM = (
  RESMIMIC_N_OBS_SINGLE * (RESMIMIC_HISTORY_LEN + 1)
  + len(RESMIMIC_FUTURE_STEPS) * (6 + 29 + 1)
)  # 1757

DEFAULT_BASE_POLICY_PATH = Path(
  "/home/jooyoungbae/ResMimic/legged_gym/assets/checkpoints/base_policy.pt"
)


def resmimic_default_joint_pos_tensor(device: torch.device | str) -> torch.Tensor:
  return torch.tensor(
    [RESMIMIC_DEFAULT_JOINT_POS[n] for n in G1_JOINT_NAMES],
    dtype=torch.float32,
    device=device,
  )


class ResMimicObsAdapter:
  """Build ResMimic student_future base obs (no object channel) from mjlab state."""

  def __init__(self, env: ManagerBasedRlEnv, command_name: str = "motion") -> None:
    self.env = env
    self.command_name = command_name
    self.device = env.device
    self.num_envs = env.num_envs
    self._default_q = resmimic_default_joint_pos_tensor(self.device)
    self._history = torch.zeros(
      self.num_envs,
      RESMIMIC_HISTORY_LEN,
      RESMIMIC_N_OBS_SINGLE,
      device=self.device,
    )
    self._last_total_action = torch.zeros(
      self.num_envs, 29, device=self.device
    )
    self._bootstrapped = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def set_last_total_action(self, actions: torch.Tensor) -> None:
    self._last_total_action = actions.detach()

  def _motion(self) -> MotionCommand:
    return self.env.command_manager.get_term(self.command_name)  # type: ignore[return-value]

  def _mimic_from_frames(self, frame_idx: torch.Tensor) -> torch.Tensor:
    """ResMimic student mimic (35): vel_xy_local, z, roll, pitch, yaw_rate, dof."""
    motion = self._motion()
    T = motion.motion.time_step_total
    idx = frame_idx.clamp(0, T - 1)
    root_pos = motion.motion.body_pos_w[idx, motion.motion_anchor_body_index]
    root_quat = motion.motion.body_quat_w[idx, motion.motion_anchor_body_index]
    root_lin_vel = motion.motion.body_lin_vel_w[idx, motion.motion_anchor_body_index]
    root_ang_vel = motion.motion.body_ang_vel_w[idx, motion.motion_anchor_body_index]
    dof_pos = motion.motion.joint_pos[idx]

    root_vel_local = quat_apply_inverse(root_quat, root_lin_vel)
    root_ang_vel_local = quat_apply_inverse(root_quat, root_ang_vel)
    roll, pitch, _ = euler_xyz_from_quat(root_quat)
    return torch.cat(
      (
        root_vel_local[:, :2],
        root_pos[:, 2:3],
        roll.unsqueeze(-1),
        pitch.unsqueeze(-1),
        root_ang_vel_local[:, 2:3],
        dof_pos,
      ),
      dim=-1,
    )

  def _proprio(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    root_quat = robot.data.root_link_quat_w
    root_ang_vel_w = robot.data.root_link_ang_vel_w
    base_ang_vel = quat_apply_inverse(root_quat, root_ang_vel_w)
    roll, pitch, _ = euler_xyz_from_quat(root_quat)
    imu = torch.stack((roll, pitch), dim=-1)
    dof_pos = robot.data.joint_pos - self._default_q.unsqueeze(0)
    dof_vel = robot.data.joint_vel
    # Zero ankle dof vel like ResMimic (indices 4,5,10,11).
    dof_vel = dof_vel.clone()
    dof_vel[:, [4, 5, 10, 11]] = 0.0
    return torch.cat(
      (base_ang_vel, imu, dof_pos, dof_vel, self._last_total_action), dim=-1
    )

  def _current_single(self) -> torch.Tensor:
    motion = self._motion()
    mimic = self._mimic_from_frames(motion.time_steps)
    proprio = self._proprio()
    return torch.cat((mimic, proprio), dim=-1)

  def _future(self) -> torch.Tensor:
    motion = self._motion()
    chunks = []
    for step in RESMIMIC_FUTURE_STEPS:
      frames = motion.time_steps + step
      mimic = self._mimic_from_frames(frames)
      # mask indicator = 1 (visible); no curriculum masking in this port
      mask = torch.ones(self.num_envs, 1, device=self.device)
      chunks.append(torch.cat((mimic, mask), dim=-1))
    return torch.cat(chunks, dim=-1)

  def reset_(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids.numel() == 0:
      return
    cur = self._current_single()[env_ids]
    self._history[env_ids] = cur.unsqueeze(1).expand(-1, RESMIMIC_HISTORY_LEN, -1)
    self._bootstrapped[env_ids] = True
    self._last_total_action[env_ids] = 0.0

  def build(self, update_history: bool = True) -> torch.Tensor:
    """Return (num_envs, 1757) base obs."""
    # Freshly reset envs: fill history with current.
    if hasattr(self.env, "episode_length_buf"):
      just_reset = self.env.episode_length_buf <= 1
      if just_reset.any():
        self.reset_(just_reset.nonzero(as_tuple=False).squeeze(-1))

    current = self._current_single()
    if update_history:
      # shift left, append previous current handled by rolling after obs use:
      # ResMimic stores history of past currents; we append *previous* by
      # shifting and writing current at the end after build for next step.
      pass

    hist_flat = self._history.reshape(self.num_envs, -1)
    future = self._future()
    obs = torch.cat((current, hist_flat, future), dim=-1)
    assert obs.shape[-1] == RESMIMIC_OBS_DIM, obs.shape

    if update_history:
      # Push current into history ring (oldest dropped).
      self._history = torch.cat(
        (self._history[:, 1:], current.unsqueeze(1)), dim=1
      )
    return obs


class FrozenResMimicBasePolicy:
  """TorchScript ResMimic base + conversion into mjlab raw action space."""

  def __init__(
    self,
    policy_path: str | Path,
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
  ) -> None:
    self.device = env.device
    path = Path(policy_path).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"ResMimic base policy not found: {path}")
    self.policy = torch.jit.load(str(path), map_location=self.device)
    self.policy.eval()
    self.adapter = ResMimicObsAdapter(env, command_name=command_name)

    action_term = env.action_manager.get_term("joint_pos")
    scale = action_term._scale
    if isinstance(scale, float):
      self._scale = torch.full(
        (1, 29), float(scale), device=self.device, dtype=torch.float32
      )
    else:
      self._scale = scale[:1].to(self.device)

    offset = action_term._offset
    if isinstance(offset, float):
      # use_default_offset normally makes offset = default_joint_pos
      self._mj_default = env.scene["robot"].data.default_joint_pos[:1].to(self.device)
    else:
      self._mj_default = offset[:1].to(self.device)

    self._res_default = resmimic_default_joint_pos_tensor(self.device).unsqueeze(0)

  @torch.no_grad()
  def act_mjlab_raw(self, update_history: bool = True) -> torch.Tensor:
    """Return base contribution in mjlab *raw* action units (before scale)."""
    base_obs = self.adapter.build(update_history=update_history)
    a_res = self.policy(base_obs)
    # q_target = res_default + 0.5 * a_res = mj_default + scale * a_mj
    q_target = self._res_default + RESMIMIC_ACTION_SCALE * a_res
    a_mj = (q_target - self._mj_default) / self._scale.clamp_min(1e-6)
    return a_mj
