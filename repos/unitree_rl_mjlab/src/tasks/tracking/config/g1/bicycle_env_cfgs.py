"""Unitree G1 + GMR bicycle tracking environment configurations."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.mdp.bicycle_psi import BicyclePSIMotionCommandCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_dhand_robot_cfg, get_g1_robot_cfg
from src.assets.robots.g1_bicycle import get_g1_bicycle_cfg
from src.assets.robots.g1_bicycle.bicycle_constants import (
  BICYCLE_SADDLE58_XML,
  BICYCLE_SADDLE60_XML,
)
from src.tasks.tracking.mdp.bicycle_actions import BicycleReferenceActionCfg
from src.tasks.tracking.mdp.bicycle_data import SADDLE58_LOCAL_POS, SADDLE60_LOCAL_POS
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .bicycle_mdp import (
  add_bicycle_curriculum,
  add_bicycle_depth_grid_obs,
  add_bicycle_freewheel,
  add_bicycle_observations,
  add_pedal_net_force_observation,
  add_bicycle_rewards,
  add_bike_interaction_local_points,
  add_pelvis_saddle_lock_curriculum,
  add_proprio_history,
  add_rich_actor_pose_and_heights,
  add_t084_foot_ground_penalty,
  configure_bike052_rewards,
  configure_contact_30_20_50_no_backward,
  configure_crank_progress_top_pedal_rewards,
  configure_deployable_lesson_rewards,
  configure_deployable_hist5_obs,
  configure_bicycle_distill_observations,
  configure_bicycle_dual_privileged_latent_observations,
  configure_pedal_window_slim_rewards,
  configure_physics_pd_actor_obs,
  configure_physics_pd_rewards,
  configure_ratchet_ablation_rewards,
  configure_ratchet_v3_dr_curriculum,
  configure_hybrid_dm_amp_ride20s,
  configure_resmimic_residual_actor_obs_contact,
  configure_legacy_hist9_rich_1490_obs,
  configure_legacy_resmimic_988_obs,
  configure_slim_bike_obs,
  disable_actor_observation_noise,
  disable_general_domain_randomization,
  make_bicycle_contact_sensors,
  make_pedal_net_force_sensors,
  make_ratchet_constraint_sensors,
  make_feet_ground_sensor,
)

DEFAULT_PELVIS_LOCK_PEDAL_MOTION = (
  "src/assets/motions/g1/"
  "bike_052_f375-440_pelvis_saddle_lock_50hz_dhand.npz"
)

DEFAULT_WRIST_PEDAL_FULL_MOTION = (
  "src/assets/motions/g1/"
  "bike_052_holosoma_wrist_pedal_full_50hz_dhand_saddle60.npz"
)

DEFAULT_SOLE_STANCESPREAD_MOTION = (
  "src/assets/motions/g1/"
  "bike_052_holosoma_wrist_pedal_sole_stancespread2_spread2fwd3_"
  "full_50hz_dhand_saddle60.npz"
)

DEFAULT_HYBRID_RIDE20_MOTION = (
  "src/assets/motions/g1/"
  "bike_052_hybrid_dm_amp_ride20s_cmd2_relgroundfix_50hz_dhand_saddle60.npz"
)

DEFAULT_WRIST_PEDAL_SADDLE58_PEDAL10_MOTION = (
  "src/assets/motions/g1/"
  "bike_052_holosoma_wrist_pedal_full_50hz_dhand_saddle58_pedal10.npz"
)

DEFAULT_BICYCLE_MOTION = (
  "src/assets/motions/g1/test_motion1_t084_holosoma_v1_g1_bicycle_mjlab_with_bike.npz"
)

DEFAULT_BIKE052_MOTION = (
  "src/assets/motions/g1/bike_052_c7_saddle60_g1_bicycle_mjlab_with_bike.npz"
)

CurriculumMode = Literal["A", "B", "C", "bike052", "PD"]
RobotHand = Literal["rubber", "dhand"]


def unitree_g1_bicycle_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  foot_ground_penalty: bool = False,
  curriculum_mode: CurriculumMode | None = None,
  raycast_depth_grid: bool = False,
  proprio_history_length: int = 0,
  enhanced_bike_local_points: bool = False,
  rich_actor_obs: bool = False,
  ccd_iterations: int | None = None,
  robot_hand: RobotHand = "rubber",
  bicycle_xml: Path | None = None,
  saddle_local_pos: tuple[float, float, float] | None = None,
  default_motion_file: str | None = None,
  physics_pd_decay_until_iter: int = 50_000,
  physics_pd_assist_levels: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5, 1.0),
  pedal_contact_details: bool = False,
  include_pedal_net_force_obs: bool = False,
  dhand_hand_friction: float = 1.5,
  bicycle_collision_condim: int | None = None,
  handlebar_friction: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """G1 motion tracking with a GMR bicycle in the scene.

  ``curriculum_mode``:
    - ``None``: kinematic bicycle driven from motion.npz each step (legacy).
    - ``"A"``: dynamic bike + virtual PD stabilization curriculum.
    - ``"B"``: dynamic bike + partial kinematic lock curriculum.
    - ``"C"``: dynamic freewheel bike; decaying roll anti-fall + timed forward push.
    - ``"bike052"``: dynamic freewheel; 30k decaying roll hold + pelvis +X push (11–15 s).

  ``proprio_history_length``:
    If > 0, stack that many past steps on proprio terms (joint/action/IMU).
  ``enhanced_bike_local_points``:
    Add explicit handle / saddle / pedal points in the robot-anchor frame.
  ``rich_actor_obs``:
    Keep/add motion_anchor_pos_b, base_lin_vel, body vel, and explicit
    pelvis/torso/feet height channels. Drops body_pos/ori (FK⊂joints).
  ``ccd_iterations``:
    Override MuJoCo CCD iterations (default: 50 kin / 500 curriculum).
  ``robot_hand``:
    ``"rubber"`` (default capsule hand) or ``"dhand"`` (ㄷ 4-box grip).
  ``bicycle_xml`` / ``saddle_local_pos``:
    Override bike asset and saddle interaction point (e.g. saddle60).
  """
  dynamic_bike = curriculum_mode is not None
  cfg = make_tracking_env_cfg()

  robot_cfg = (
    get_g1_dhand_robot_cfg(hand_friction=dhand_hand_friction)
    if robot_hand == "dhand"
    else get_g1_robot_cfg()
  )
  cfg.scene.entities = {
    "robot": robot_cfg,
    "bicycle": get_g1_bicycle_cfg(
      dynamic=dynamic_bike,
      freewheel=dynamic_bike,
      xml_path=bicycle_xml,
      collision_condim=bicycle_collision_condim,
      handlebar_friction=handlebar_friction,
    ),
  }

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  bicycle_sensors = make_bicycle_contact_sensors(
    dhand_inner_handle=(robot_hand == "dhand"),
    pedal_contact_details=pedal_contact_details,
  )
  pedal_force_sensors = (
    make_pedal_net_force_sensors() if include_pedal_net_force_obs else ()
  )
  if foot_ground_penalty:
    cfg.scene.sensors = (
      self_collision_cfg,
      make_feet_ground_sensor(dhand_sole=(robot_hand == "dhand")),
      *bicycle_sensors,
      *pedal_force_sensors,
    )
  else:
    cfg.scene.sensors = (
      self_collision_cfg,
      *bicycle_sensors,
      *pedal_force_sensors,
    )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  if dynamic_bike:
    cfg.actions.pop("bicycle_reference", None)
  else:
    cfg.actions["bicycle_reference"] = BicycleReferenceActionCfg(
      entity_name="bicycle",
      command_name="motion",
    )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = default_motion_file or DEFAULT_BICYCLE_MOTION
  motion_cmd.bicycle_entity_name = "bicycle"
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7](_sole)?_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  add_bicycle_observations(cfg)
  if include_pedal_net_force_obs:
    add_pedal_net_force_observation(cfg, force_scale_n=100.0)
  if enhanced_bike_local_points:
    add_bike_interaction_local_points(cfg, saddle_local_pos=saddle_local_pos)
  if rich_actor_obs:
    add_rich_actor_pose_and_heights(cfg)
  if proprio_history_length > 0:
    add_proprio_history(cfg, history_length=proprio_history_length)
  if raycast_depth_grid:
    add_bicycle_depth_grid_obs(cfg)
  handle_min_plates = 2 if robot_hand == "dhand" else 1
  add_bicycle_rewards(
    cfg,
    curriculum=dynamic_bike,
    handle_min_inner_plates=handle_min_plates,
  )
  if foot_ground_penalty:
    add_t084_foot_ground_penalty(cfg)
  if curriculum_mode is not None:
    if curriculum_mode == "bike052":
      add_bicycle_curriculum(
        cfg,
        mode="bike052",
        decay_until_iter=30_000,
        pelvis_push_target_speed=1.0,
        pelvis_push_start_frame=550,
        pelvis_push_end_frame=750,
        saddle_contact_local=SADDLE60_LOCAL_POS,
      )
      configure_bike052_rewards(cfg, handle_min_inner_plates=handle_min_plates)
      disable_actor_observation_noise(cfg)
    elif curriculum_mode == "PD":
      add_bicycle_curriculum(
        cfg,
        mode="PD",
        decay_until_iter=physics_pd_decay_until_iter,
        saddle_contact_local=saddle_local_pos,
        pd_assist_levels=physics_pd_assist_levels,
      )
    else:
      add_bicycle_curriculum(cfg, mode=curriculum_mode)
    add_bicycle_freewheel(cfg)
    cfg.sim.nconmax = 128
    # Dynamic bike + full collision geoms can exceed default njmax=250.
    cfg.sim.njmax = 384
    cfg.sim.mujoco.ccd_iterations = 500
  else:
    cfg.sim.nconmax = 80
    # Kinematic bike still needs headroom (dhand sole pads + pedal contacts).
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 384)

  if ccd_iterations is not None:
    cfg.sim.mujoco.ccd_iterations = ccd_iterations
    # Kin Hist9 / anti-NaN buffer: raise contact budgets without full CCD=500.
    if not dynamic_bike:
      cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
      cfg.sim.njmax = max(cfg.sim.njmax or 0, 384)

  if not has_state_estimation and not rich_actor_obs:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # No general BeyondMimic DR for bike runs (RSI pose/vel/joint, CoM,
  # friction, encoder bias, push, obs noise). Curriculum assists stay.
  disable_general_domain_randomization(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_bicycle_physics_pd_env_cfg(
  *,
  play: bool = False,
  saddle: Literal["60", "58_pedal10"] = "60",
  include_acceleration: bool = False,
  robot_hand: RobotHand = "dhand",
  crank_progress_top_pedal: bool = False,
  contact_30_20_50_no_backward: bool = False,
  assistance_mode: Literal["PD", "B"] = "PD",
  include_pedal_net_force_obs: bool = False,
  dhand_hand_friction: float = 1.5,
  bicycle_collision_condim: int | None = None,
  handlebar_friction: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Dynamic bike task with matched PD or Curr-B assistance curricula."""
  if saddle == "60":
    bicycle_xml = BICYCLE_SADDLE60_XML
    saddle_local = SADDLE60_LOCAL_POS
    motion_file = DEFAULT_WRIST_PEDAL_FULL_MOTION
  else:
    bicycle_xml = BICYCLE_SADDLE58_XML
    saddle_local = SADDLE58_LOCAL_POS
    motion_file = DEFAULT_WRIST_PEDAL_SADDLE58_PEDAL10_MOTION

  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode=assistance_mode,
    robot_hand=robot_hand,
    bicycle_xml=bicycle_xml,
    saddle_local_pos=saddle_local,
    default_motion_file=motion_file,
    physics_pd_decay_until_iter=50_000,
    physics_pd_assist_levels=(0.0, 0.05, 0.2, 0.5, 1.0),
    pedal_contact_details=crank_progress_top_pedal,
    include_pedal_net_force_obs=include_pedal_net_force_obs,
    dhand_hand_friction=dhand_hand_friction,
    bicycle_collision_condim=bicycle_collision_condim,
    handlebar_friction=handlebar_friction,
  )
  if assistance_mode == "B":
    # Matched 50k comparison: alpha blends the complete bicycle root/joint
    # state from kinematic reference (1) to unconstrained physics (0).
    init_cfg = cfg.events["bicycle_curriculum_init"]
    init_cfg.params.update(
      {
        "decay_until_iter": 50_000,
        "saddle_contact_local": saddle_local,
      }
    )
  cfg.episode_length_s = int(1e9) if play else 22.0
  # At 2048 worlds the initial random policy can create dense simultaneous
  # robot/bike/self contacts.  The observed peak required nefc was 461; leave
  # ample headroom so MuJoCo-Warp does not silently truncate constraints.
  cfg.sim.njmax = 768
  # Seven sole layers per foot increase the number of geom-pair matches seen
  # by the contact sensors.  Runtime initialization observed up to 85 matches;
  # keep margin so sole/pedal contacts are not silently dropped.
  cfg.sim.contact_sensor_maxmatch = 128
  # The full-physics task initially deviates more than the kinematic curriculum.
  # Keep the tolerance used by the prior successful bike runs so an untrained
  # policy receives enough trajectory before termination to learn recovery.
  cfg.terminations["anchor_pos"].params["threshold"] = 0.4
  # Full-clip RSI often starts in a seated/contact pose.  A random initial
  # action can move a wrist/ankle vertically without the robot having fallen;
  # terminating on that limb alone produced ~2-step rollouts and no learnable
  # trajectory.  Root height/orientation still terminate genuine falls, while
  # body tracking rewards continuously penalize limb errors.
  cfg.terminations.pop("ee_body_pos", None)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.convert_bike_qvel_from_qpos_derivative = True
  configure_physics_pd_actor_obs(
    cfg,
    saddle_local_pos=saddle_local,
    include_acceleration=include_acceleration,
  )
  # The specialized PhysicsPD observation builder replaces the base groups,
  # so append this opt-in term only after that replacement.
  if include_pedal_net_force_obs:
    add_pedal_net_force_observation(cfg, force_scale_n=100.0)
  configure_physics_pd_rewards(cfg)
  if crank_progress_top_pedal:
    configure_crank_progress_top_pedal_rewards(cfg)
  if contact_30_20_50_no_backward:
    configure_contact_30_20_50_no_backward(cfg)
  disable_actor_observation_noise(cfg)
  return cfg


def unitree_g1_bicycle_ratchet_ablation_env_cfg(
  *,
  play: bool = False,
  variant: Literal[
    "v1_term",
    "v1_dist10",
    "v2_term_lower",
    "v3_backward2",
    "v4_backward1_lower",
  ],
) -> ManagerBasedRlEnvCfg:
  """Four controlled ratcheting experiments derived from the best PD run."""
  variant_settings = {
    "v1_term": (True, False, None, False),
    "v1_dist10": (False, True, None, False),
    "v2_term_lower": (True, False, None, True),
    "v3_backward2": (False, False, 2.0, False),
    "v4_backward1_lower": (False, False, 1.0, True),
  }
  (
    pedal_miss_termination,
    pedal_distance_termination,
    backward_weight,
    lower_body_half,
  ) = variant_settings[variant]

  # Start from the stored 2026-08-17 PhysicsPD structure. Keep the legacy
  # foot1--7 netforce pedal sensors; crank progress is added explicitly below.
  cfg = unitree_g1_bicycle_physics_pd_env_cfg(
    play=play,
    saddle="60",
    include_acceleration=False,
    robot_hand="dhand",
    crank_progress_top_pedal=False,
    contact_30_20_50_no_backward=False,
    assistance_mode="PD",
    dhand_hand_friction=1.0,
    bicycle_collision_condim=6,
    handlebar_friction=1.0,
  )
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = DEFAULT_SOLE_STANCESPREAD_MOTION

  # Common updated interaction geometry and safety constraints.
  cfg.scene.sensors = (cfg.scene.sensors or ()) + make_ratchet_constraint_sensors()
  configure_crank_progress_top_pedal_rewards(cfg)
  configure_ratchet_ablation_rewards(
    cfg,
    backward_weight=backward_weight,
    lower_body_half=lower_body_half,
    pedal_miss_termination=pedal_miss_termination,
    pedal_distance_termination=pedal_distance_termination,
  )
  return cfg


def unitree_g1_bicycle_ratchet_distill_env_cfg(
  *,
  play: bool = False,
  variant: Literal[
    "v1_term",
    "v1_dist10",
    "v2_term_lower",
    "v3_backward2",
    "v4_backward1_lower",
  ] = "v3_backward2",
) -> ManagerBasedRlEnvCfg:
  """Ratchet teacher dynamics/rewards with a deployable student actor input.

  Rewards, terminations, contacts, PhysicsPD curriculum, and simulator settings
  come from the selected teacher task without modification. Only observation
  routing changes: privileged teacher, deployable student actor, privileged
  asymmetric critic.
  """
  cfg = unitree_g1_bicycle_ratchet_ablation_env_cfg(play=play, variant=variant)
  configure_bicycle_distill_observations(cfg, history_length=5)
  return cfg


def unitree_g1_bicycle_ratchet_v3_dual_latent_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Exact Ratchet V3 environment with dual privileged actor encoders."""
  cfg = unitree_g1_bicycle_ratchet_ablation_env_cfg(
    play=play, variant="v3_backward2"
  )
  configure_bicycle_dual_privileged_latent_observations(cfg)
  return cfg


def unitree_g1_bicycle_ratchet_v3_dr_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Ratchet V3 with staged sim-to-real domain randomization."""
  cfg = unitree_g1_bicycle_ratchet_ablation_env_cfg(
    play=play, variant="v3_backward2"
  )
  if not play:
    configure_ratchet_v3_dr_curriculum(cfg)
  return cfg


def unitree_g1_bicycle_ratchet_v3_longref_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Exact Ratchet V3 teacher setup with the corrected 20 s riding reference.

  This is a DeepMimic-only control arm for the Hybrid AMP experiment.  It does
  not add the AMP discriminator/reward, speed command observation, or the 20%
  ride-phase tracking scale.  Only the motion and matching episode duration
  differ from Ratchet V3-Backward2.
  """
  cfg = unitree_g1_bicycle_ratchet_ablation_env_cfg(
    play=play, variant="v3_backward2"
  )
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = DEFAULT_HYBRID_RIDE20_MOTION
  cfg.episode_length_s = 1957 / 50.0
  return cfg


def unitree_g1_bicycle_hybrid_amp_ride20_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """V3 transitions with a 20 s conditional-AMP middle riding phase."""
  cfg = unitree_g1_bicycle_ratchet_ablation_env_cfg(
    play=play, variant="v3_backward2"
  )
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = DEFAULT_HYBRID_RIDE20_MOTION
  cfg.episode_length_s = 1957 / 50.0
  configure_hybrid_dm_amp_ride20s(cfg, training=not play)
  return cfg


def unitree_g1_bicycle_physics_pd_psi_future_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Future-use PhysicsPD task with opt-in bicycle-aware PSI.

  This is deliberately a new task factory.  Existing PhysicsPD/Curr-B command
  classes and reset behavior remain untouched.  During play PSI is disabled,
  because an in-memory training buffer is neither present nor checkpointed.
  """
  cfg = unitree_g1_bicycle_physics_pd_env_cfg(
    play=play,
    saddle="60",
    include_acceleration=False,
    robot_hand="dhand",
    crank_progress_top_pedal=True,
    contact_30_20_50_no_backward=True,
    assistance_mode="PD",
  )
  base = cfg.commands["motion"]
  assert isinstance(base, MotionCommandCfg)
  base.motion_file = DEFAULT_SOLE_STANCESPREAD_MOTION
  inherited = {
    field.name: getattr(base, field.name) for field in fields(MotionCommandCfg)
  }
  cfg.commands["motion"] = BicyclePSIMotionCommandCfg(
    **inherited,
    enabled=not play,
    physical_buffer_slots=2,
    reference_sampling_probability=0.34,
    validation_horizon_steps=50,  # 1.0 s at the 50 Hz policy rate.
    min_confirmations=2,
    max_assistance_scale_for_physical_states=0.2,
    required_contact_duty=0.8,
    min_forward_crank_progress_rad=0.05,
    handle_min_inner_plates=2,
    pedal_sole_contact=True,
  )
  return cfg


def unitree_g1_bicycle_deploy_hist5_env_cfg(
  *,
  play: bool = False,
  include_bicycle_state: bool = False,
  include_future_reference: bool = False,
  include_interaction_geometry: bool = False,
  fixed_roll_pd: bool = False,
  fixed_steer_pd: bool = False,
  include_pedal_net_force_obs: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Saddle60 deployable-observation ablation with latest contact rewards.

  ``fixed_roll_pd=False`` uses reference-driven kinematic bicycle playback.
  ``fixed_roll_pd=True`` keeps the bicycle dynamic/freewheel and applies strong
  full-scale physical roll PD for the entire 50k run.
  """
  if fixed_steer_pd and not fixed_roll_pd:
    raise ValueError("fixed_steer_pd requires fixed_roll_pd")

  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    curriculum_mode="PD" if fixed_roll_pd else None,
    robot_hand="dhand",
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_WRIST_PEDAL_FULL_MOTION,
    # Effectively constant over a 50k run; all environments use full assist.
    physics_pd_decay_until_iter=1_000_000_000_000,
    physics_pd_assist_levels=(1.0,),
    # Required for exact foot2-6 + heel-toe 50--90% pedal reward gating.
    pedal_contact_details=True,
    include_pedal_net_force_obs=include_pedal_net_force_obs,
  )

  cfg.episode_length_s = int(1e9) if play else 22.0
  cfg.terminations["anchor_pos"].params["threshold"] = 0.4
  cfg.terminations.pop("ee_body_pos", None)
  # condim=3 capacity audit (2026-08-19): the earlier 64-contact audit passed,
  # but the detailed sole maxforce sensor smoke rollout observed a world
  # requiring 66 contacts.  Keep modest headroom at 80 for 4096-env training.
  # A single play world can exceed that (narrowphase asked ~100), so raise
  # play nconmax or Warp silently drops contacts.  Observed nefc peak reached
  # 754 in the 4096-env detailed-sole rollout, so keep njmax 768.
  cfg.sim.nconmax = 128 if play else 80
  cfg.sim.njmax = 768
  # Detailed maxforce pedal sensing observed 184 simultaneous sensor matches
  # across 4096 worlds.  256 avoids silent reward-contact truncation.
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.ccd_iterations = 100

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.convert_bike_qvel_from_qpos_derivative = True

  configure_deployable_hist5_obs(
    cfg,
    include_bicycle_state=include_bicycle_state,
    include_future_reference=include_future_reference,
    include_interaction_geometry=include_interaction_geometry,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    history_length=5,
  )
  # The deployable observation builder replaces the base groups, so append
  # the force vector afterward. Existing registered tasks keep this disabled.
  if include_pedal_net_force_obs:
    add_pedal_net_force_observation(cfg, force_scale_n=100.0)
  # Keep the exact latest PhysicsPD/NoBackward reward family for a controlled
  # observation and assistance ablation, even though kinematic bike-dynamics
  # tracking terms are constant with respect to the robot policy.
  configure_physics_pd_rewards(cfg)
  configure_crank_progress_top_pedal_rewards(cfg)
  configure_contact_30_20_50_no_backward(cfg)
  configure_deployable_lesson_rewards(cfg, fixed_roll_pd=fixed_roll_pd)
  disable_actor_observation_noise(cfg)

  if fixed_roll_pd:
    # Full-scale roll assistance for all environments and all 50k iterations.
    # These stronger gains follow the earlier bicycle roll-hold setting while
    # retaining a bounded physical torque rather than overwriting bike state.
    step_cfg = cfg.events["bicycle_curriculum_step"]
    step_cfg.params.update(
      {"kp_roll": 300.0, "kd_roll": 30.0, "max_roll_torque": 120.0}
    )
    if fixed_steer_pd:
      step_cfg.params.update(
        {
          "steer_pd_enabled": True,
          "kp_steer": 40.0,
          "kd_steer": 5.0,
          "max_steer_torque": 15.0,
        }
      )
  return cfg


def unitree_g1_bike052_hist9_rich_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "rubber",
) -> ManagerBasedRlEnvCfg:
  """bike_052 C7 + saddle60 Hist9-Rich with dynamic-bike warmup assists.

  - Actor obs noise off.
  - Contact reward targets = handle hand-mids / pedal box centers / saddle60.
  - First 30k iters: roll upright assist + pelvis world-+X push (11–15 s, 1→0 m/s).
  - Pedals/wheels physical (freewheel); no kinematic bike rewrite.
  """
  return unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=True,
    rich_actor_obs=True,
    # Dynamic bike052 uses curriculum CCD=500 (do not override with kin Hist9=100).
    ccd_iterations=None,
    curriculum_mode="bike052",
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_BIKE052_MOTION,
  )


def unitree_g1_bike052_hist9_rich_curr_b_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
) -> ManagerBasedRlEnvCfg:
  """Hist9-Rich Dhand + saddle60 with Curr-B kinematic→physics unlock.

  Same actor obs as Bike052-Dhand (Hist9 + rich + enhanced bike points).
  Curriculum B: full kinematic bike lock → linear unlock to free physics.
  Decay length via env ``BICYCLE_CURRICULUM_DECAY_UNTIL_ITER`` (default 100k).
  """
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=True,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode="B",
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_BIKE052_MOTION,
  )
  configure_slim_bike_obs(cfg, saddle_local_pos=SADDLE60_LOCAL_POS, include_ori=True)
  disable_actor_observation_noise(cfg)
  configure_bike052_rewards(
    cfg, handle_min_inner_plates=(2 if robot_hand == "dhand" else 1)
  )
  return cfg


def unitree_g1_legacy_hist9_rich_curr_b_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
) -> ManagerBasedRlEnvCfg:
  """Play-only legacy: Hist9-Rich 1490D obs (pre-slim) for old Curr-B ckpts."""
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=False,
    ccd_iterations=None,
    curriculum_mode="B",
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_BIKE052_MOTION,
  )
  configure_legacy_hist9_rich_1490_obs(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS
  )
  disable_actor_observation_noise(cfg)
  return cfg


def unitree_g1_legacy_resmimic_988_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
  pelvis_lock: bool = False,
  pelvis_lock_decay_iters: float = 30000.0,
  motion_file: str | None = None,
) -> ManagerBasedRlEnvCfg:
  """Play-only legacy: ResMimic actor 988D / critic 1466D (bike_component_pos)."""
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode=None,
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=motion_file or DEFAULT_PELVIS_LOCK_PEDAL_MOTION,
  )
  if not play:
    cfg.episode_length_s = 2.5
  configure_legacy_resmimic_988_obs(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS
  )
  configure_pedal_window_slim_rewards(cfg)
  if pelvis_lock:
    add_pelvis_saddle_lock_curriculum(
      cfg,
      decay_until_iter=pelvis_lock_decay_iters,
      saddle_local_pos=SADDLE60_LOCAL_POS,
    )
  disable_actor_observation_noise(cfg)
  return cfg


def unitree_g1_bike052_curr_b_resmimic_residual_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
) -> ManagerBasedRlEnvCfg:
  """Curr-B + ResMimic residual: contact points (15) + component ori (24).

  Default motion = wrist+pedal **full** clip (~22s @50Hz). Episode timeout
  stays at tracking default 10s (max episode length). Train with
  ``--resmimic-base-policy`` so the vecenv wrapper adds frozen base actions.
  Curriculum B: kinematic bike lock → linear unlock to free physics.
  """
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,  # keep anchor pos / lin_vel for hist terms
    ccd_iterations=None,
    curriculum_mode="B",
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_WRIST_PEDAL_FULL_MOTION,
  )
  # Explicit max episode length (tracking default is already 10s).
  if not play:
    cfg.episode_length_s = 10.0
  configure_resmimic_residual_actor_obs_contact(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS, include_ori=True
  )
  disable_actor_observation_noise(cfg)
  configure_bike052_rewards(
    cfg, handle_min_inner_plates=(2 if robot_hand == "dhand" else 1)
  )
  return cfg


def unitree_g1_pedal_window_pelvis_lock_resmimic_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
  pelvis_lock_decay_iters: float = 30000.0,
) -> ManagerBasedRlEnvCfg:
  """Pedal-window POC: kinematic bike + pelvis→saddle lock (30k unlock).

  - Motion: f375-440 with pelvis locked to saddle@375 in the reference.
  - Bike: fully kinematic (``curriculum_mode=None``).
  - Pelvis: hard-locked to saddle, linear unlock over ``pelvis_lock_decay_iters``.
  - Rewards: robot tracking + eef contact only.
  - Actor: slim Hist9 + contact points (15) + ori (24); train with
    ``--resmimic-base-policy`` (GMT/ResMimic frozen base).
  - Collision: ``g1_dhand.xml`` (sideways pelvis cylinder).
  """
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode=None,  # kinematic bike
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=DEFAULT_PELVIS_LOCK_PEDAL_MOTION,
  )
  if not play:
    cfg.episode_length_s = 2.5  # cut ≈ 2.2s
  configure_resmimic_residual_actor_obs_contact(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS, include_ori=True
  )
  configure_pedal_window_slim_rewards(cfg)
  add_pelvis_saddle_lock_curriculum(
    cfg,
    decay_until_iter=pelvis_lock_decay_iters,
    saddle_local_pos=SADDLE60_LOCAL_POS,
  )
  disable_actor_observation_noise(cfg)
  return cfg


def unitree_g1_pedal_window_nolock_resmimic_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
  motion_file: str | None = None,
  episode_length_s: float | None = 2.5,
) -> ManagerBasedRlEnvCfg:
  """Same as pelvis-lock pedal POC, but no sim pelvis→saddle hard lock.

  Fair A/B vs ``unitree_g1_pedal_window_pelvis_lock_resmimic_env_cfg``:
  same motion cut, kinematic bike, slim rewards, residual obs (15+24 bike).
  """
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode=None,  # kinematic bike
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=motion_file or DEFAULT_PELVIS_LOCK_PEDAL_MOTION,
  )
  if not play and episode_length_s is not None:
    cfg.episode_length_s = float(episode_length_s)
  configure_resmimic_residual_actor_obs_contact(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS, include_ori=True
  )
  configure_pedal_window_slim_rewards(cfg)
  configure_bike052_rewards(
    cfg, handle_min_inner_plates=(2 if robot_hand == "dhand" else 1)
  )
  disable_actor_observation_noise(cfg)
  return cfg


def unitree_g1_pedal_nolock_curr_b_resmimic_env_cfg(
  *,
  play: bool = False,
  robot_hand: RobotHand = "dhand",
  motion_file: str | None = None,
  episode_length_s: float | None = 22.0,
) -> ManagerBasedRlEnvCfg:
  """PedalNoLock ResMimic residual + Curr-B kinematic→physics unlock.

  Same slim obs/rewards as ``PedalNoLock-ResMimicResidual`` (no bike pose
  tracking terms). Decay length via ``BICYCLE_CURRICULUM_DECAY_UNTIL_ITER``.
  """
  cfg = unitree_g1_bicycle_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
    proprio_history_length=9,
    enhanced_bike_local_points=False,
    rich_actor_obs=True,
    ccd_iterations=None,
    curriculum_mode="B",
    robot_hand=robot_hand,
    bicycle_xml=BICYCLE_SADDLE60_XML,
    saddle_local_pos=SADDLE60_LOCAL_POS,
    default_motion_file=motion_file or DEFAULT_WRIST_PEDAL_FULL_MOTION,
  )
  if not play and episode_length_s is not None:
    cfg.episode_length_s = float(episode_length_s)
  configure_resmimic_residual_actor_obs_contact(
    cfg, saddle_local_pos=SADDLE60_LOCAL_POS, include_ori=True
  )
  configure_pedal_window_slim_rewards(cfg)
  configure_bike052_rewards(
    cfg, handle_min_inner_plates=(2 if robot_hand == "dhand" else 1)
  )
  disable_actor_observation_noise(cfg)
  return cfg
