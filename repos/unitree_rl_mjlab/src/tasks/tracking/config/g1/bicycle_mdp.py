"""Wire bicycle obs/rewards/sensors into the G1 bicycle tracking env cfg."""

from __future__ import annotations

import copy
from typing import Literal

from mjlab.actuator import DelayedActuatorCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
import src.tasks.tracking.mdp as mdp
from src.tasks.tracking.amp import AmpCfg
from src.tasks.tracking.mdp.bicycle_data import (
  BICYCLE_CONTACT_PAIRS,
  BICYCLE_EEF_CONTACT_PAIR_WEIGHTS,
  BICYCLE_OBS_BODIES,
  BICYCLE_SPLIT_CONTACT_KEYS,
  BICYCLE_SPLIT_CONTACT_REWARD_NAMES,
)

BICYCLE_DEPTH_GRID_SENSOR = "bike_depth_grid"
BICYCLE_DEPTH_GRID_MAX_DISTANCE = 2.0
PEDAL_NET_FORCE_SENSOR_NAMES = {
  "left": "contact_pedal_left_netforce",
  "right": "contact_pedal_right_netforce",
}
FORBIDDEN_BIKE_CONTACT_SENSOR = "forbidden_robot_bicycle_contact"
HANDLE_DETACH_SENSOR_NAMES = {
  "handle_left": "contact_handle_left_all_plates",
  "handle_right": "contact_handle_right_all_plates",
}


def make_bicycle_depth_grid_sensor() -> RayCastSensorCfg:
  """Pelvis-mounted 0.7 x 0.7 m downward ray grid (yaw-aligned)."""
  return RayCastSensorCfg(
    name=BICYCLE_DEPTH_GRID_SENSOR,
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(
      size=(0.7, 0.7),
      resolution=0.1,
      direction=(0.0, 0.0, -1.0),
    ),
    max_distance=BICYCLE_DEPTH_GRID_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0, 1, 2, 3),
    debug_vis=False,
  )


def add_bicycle_depth_grid_obs(cfg) -> None:
  """Attach raycast depth grid sensor + actor/critic observation terms."""
  depth_sensor = make_bicycle_depth_grid_sensor()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_sensor,)

  depth_term = ObservationTermCfg(
    func=mdp.raycast_depth_grid,
    params={"sensor_name": BICYCLE_DEPTH_GRID_SENSOR},
    scale=1.0,
  )
  actor = dict(cfg.observations["actor"].terms)
  actor["bike_depth_grid"] = depth_term
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor,
    concatenate_terms=cfg.observations["actor"].concatenate_terms,
    enable_corruption=cfg.observations["actor"].enable_corruption,
  )

  critic = dict(cfg.observations["critic"].terms)
  critic["bike_depth_grid"] = ObservationTermCfg(
    func=mdp.raycast_depth_grid,
    params={"sensor_name": BICYCLE_DEPTH_GRID_SENSOR},
    scale=1.0,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic,
    concatenate_terms=cfg.observations["critic"].concatenate_terms,
    enable_corruption=cfg.observations["critic"].enable_corruption,
  )


def make_feet_ground_sensor(*, dhand_sole: bool = False) -> ContactSensorCfg:
  """Foot collision geoms vs terrain plane."""
  foot_pattern = (
    r"^(left|right)_foot\d+_sole_collision$"
    if dhand_sole
    else r"^(left|right)_foot\d+_collision$"
  )
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=foot_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=4,
    history_length=4,
  )


def make_bicycle_contact_sensors(
  *,
  dhand_inner_handle: bool = False,
  pedal_contact_details: bool = False,
) -> tuple[ContactSensorCfg, ...]:
  """Build EE–bike contact sensors.

  ``dhand_inner_handle=True``: handle sensors use only the thin
  cavity-facing face pads ``*_hand_plate_{back,left,right}_inner_collision``
  (not the full plate boxes / wrist).

  Pedal sensors (dhand): thin sole layers ``*_foot[1-7]_sole_collision``
  only — not the full foot boxes (sides/top do not earn pedal reward).
  """
  sensors: list[ContactSensorCfg] = []
  for pair in BICYCLE_CONTACT_PAIRS:
    if pair.target_mode == "ground":
      if dhand_inner_handle:
        ee_pattern = (
          r"left_foot\d+_sole_collision"
          if "left" in pair.key
          else r"right_foot\d+_sole_collision"
        )
      else:
        ee_pattern = (
          r"left_foot\d+_collision"
          if "left" in pair.key
          else r"right_foot\d+_collision"
        )
      sensors.append(
        ContactSensorCfg(
          name=pair.sensor_name,
          primary=ContactMatch(mode="geom", pattern=ee_pattern, entity="robot"),
          secondary=ContactMatch(mode="body", pattern="terrain"),
          fields=("found", "force"),
          reduce="netforce",
          global_frame=True,
          num_slots=4,
          history_length=4,
        )
      )
      continue

    if pair.key.startswith("handle"):
      side = "left" if "left" in pair.key else "right"
      if dhand_inner_handle:
        # One thin pad per plate, on the cavity-facing face only.
        ee_pattern = rf"{side}_hand_plate_(back|left|right)_inner_collision"
      else:
        # Rubber capsule or any hand_*_collision.
        ee_pattern = rf"{side}_hand.*_collision"
      # Updated bar has separate left/right grab segments. Use an exact
      # secondary so ContactSensor's single-secondary policy cannot silently
      # select the wrong side.
      secondary_pattern = f"handlebar_col_{side}"
      n_slots = 4
    elif pair.key == "saddle":
      ee_pattern = "pelvis_collision"
      secondary_pattern = r"saddle_\d+_col"
      n_slots = 4
    elif pair.key == "pedal_left":
      if dhand_inner_handle:
        ee_pattern = (
          r"left_foot[2-6]_sole_collision"
          if pedal_contact_details
          else r"left_foot\d+_sole_collision"
        )
        n_slots = 8
      else:
        ee_pattern = r"left_foot\d+_collision"
        n_slots = 4
      secondary_pattern = "pedal_l"
    elif pair.key == "pedal_right":
      if dhand_inner_handle:
        ee_pattern = (
          r"right_foot[2-6]_sole_collision"
          if pedal_contact_details
          else r"right_foot\d+_sole_collision"
        )
        n_slots = 8
      else:
        ee_pattern = r"right_foot\d+_collision"
        n_slots = 4
      secondary_pattern = "pedal_r"
    else:
      raise ValueError(f"Unknown contact pair key: {pair.key}")

    detailed_pedal = pedal_contact_details and pair.key.startswith("pedal_")
    fields = (
      ("found", "force", "pos")
      if detailed_pedal
      else ("found", "force")
    )
    sensors.append(
      ContactSensorCfg(
        name=pair.sensor_name,
        primary=ContactMatch(mode="geom", pattern=ee_pattern, entity="robot"),
        secondary=ContactMatch(
          mode="geom", pattern=secondary_pattern, entity="bicycle"
        ),
        fields=fields,
        # Detailed pedal sensing retains individual strongest contact slots so
        # their world positions can be tested against the sole forefoot band. Force
        # magnitude is frame invariant, so no global-frame conversion is needed.
        reduce="maxforce" if detailed_pedal else "netforce",
        global_frame=not detailed_pedal,
        num_slots=n_slots,
        history_length=4,
      )
    )
  return tuple(sensors)


def make_ratchet_constraint_sensors() -> tuple[ContactSensorCfg, ...]:
  """Sensors for forbidden robot-bike contacts and post-schedule hand detach."""
  allowed_robot_geoms = (
    r"^pelvis_collision$",
    r"^(left|right)_(hip|thigh)_collision$",
    r"^(left|right)_foot[1-7](_sole)?_collision$",
    r"^(left|right)_hand_plate_.*_collision$",
  )
  sensors: list[ContactSensorCfg] = [
    ContactSensorCfg(
      name=FORBIDDEN_BIKE_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="geom",
        pattern=r".*_collision",
        entity="robot",
        exclude=allowed_robot_geoms,
      ),
      secondary=ContactMatch(mode="subtree", pattern="frame", entity="bicycle"),
      fields=("found", "force"),
      reduce="netforce",
      global_frame=True,
      num_slots=1,
      history_length=4,
    )
  ]
  for pair_key, side in (("handle_left", "left"), ("handle_right", "right")):
    sensors.append(
      ContactSensorCfg(
        name=HANDLE_DETACH_SENSOR_NAMES[pair_key],
        primary=ContactMatch(
          mode="geom",
          pattern=rf"{side}_hand_plate_.*_collision",
          entity="robot",
        ),
        secondary=ContactMatch(
          mode="geom", pattern=f"handlebar_col_{side}", entity="bicycle"
        ),
        fields=("found", "force"),
        reduce="netforce",
        global_frame=True,
        num_slots=1,
        history_length=4,
      )
    )
  return tuple(sensors)


def make_pedal_net_force_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
  """Dedicated deployable left/right pedal load sensors.

  The detailed reward sensors retain individual max-force contact positions.
  Those vectors live in different contact frames and cannot be summed safely.
  These additional sensors instead aggregate each foot2--6 sole strip with
  ``netforce`` in the world frame; the observation then sums the five strips
  and rotates the result into the corresponding pedal frame.
  """
  sensors: list[ContactSensorCfg] = []
  for side, pedal_geom in (("left", "pedal_l"), ("right", "pedal_r")):
    sensors.append(
      ContactSensorCfg(
        name=PEDAL_NET_FORCE_SENSOR_NAMES[side],
        primary=ContactMatch(
          mode="geom",
          pattern=rf"{side}_foot[2-6]_sole_collision",
          entity="robot",
        ),
        secondary=ContactMatch(
          mode="geom", pattern=pedal_geom, entity="bicycle"
        ),
        fields=("force",),
        reduce="netforce",
        global_frame=True,
        num_slots=1,
        history_length=4,
      )
    )
  return tuple(sensors)  # type: ignore[return-value]


def add_pedal_net_force_observation(
  cfg, *, force_scale_n: float = 100.0
) -> None:
  """Add the 6-D pedal-local net-force vector to actor and critic.

  This is opt-in so existing task observation dimensions/checkpoints remain
  unchanged. A 100 N scale maps ordinary pedal loads near unit magnitude.
  """
  if force_scale_n <= 0.0:
    raise ValueError("force_scale_n must be positive")
  term = ObservationTermCfg(
    func=mdp.bicycle_pedal_net_force_local,
    params={
      "command_name": "motion",
      "left_sensor_name": PEDAL_NET_FORCE_SENSOR_NAMES["left"],
      "right_sensor_name": PEDAL_NET_FORCE_SENSOR_NAMES["right"],
      "force_scale_n": force_scale_n,
    },
    clip=(-5.0, 5.0),
  )
  for group_name in ("actor", "critic"):
    terms = dict(cfg.observations[group_name].terms)
    terms["bicycle_pedal_net_force_local"] = term
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=cfg.observations[group_name].concatenate_terms,
      enable_corruption=cfg.observations[group_name].enable_corruption,
    )


def add_bicycle_observations(cfg) -> None:
  """Append bicycle-related observation terms (privileged teacher set on actor+critic)."""
  priv_cmd = {"command_name": "motion"}
  # Teacher/student distill later: put the full privileged pack on BOTH actor and critic.
  privilege_terms: dict[str, ObservationTermCfg] = {
    # legacy coarse bike points / ori
    "bicycle_ref_points_b": ObservationTermCfg(
      func=mdp.bicycle_ref_points_b, params=priv_cmd
    ),
    "bicycle_sim_points_b": ObservationTermCfg(
      func=mdp.bicycle_sim_points_b, params=priv_cmd
    ),
    "bicycle_ref_ori_b": ObservationTermCfg(
      func=mdp.bicycle_ref_ori_b, params=priv_cmd
    ),
    "bicycle_sim_ori_b": ObservationTermCfg(
      func=mdp.bicycle_sim_ori_b, params=priv_cmd
    ),
    # per-component bike linear velocities (frame/fork/wheels/crank/pedals)
    "bicycle_ref_lin_vel_b": ObservationTermCfg(
      func=mdp.bicycle_ref_lin_vel_b, params=priv_cmd
    ),
    "bicycle_sim_lin_vel_b": ObservationTermCfg(
      func=mdp.bicycle_sim_lin_vel_b, params=priv_cmd
    ),
    # foot / pedal geometry
    "foot_pos_b": ObservationTermCfg(func=mdp.foot_pos_b, params=priv_cmd),
    "pedal_pos_b": ObservationTermCfg(func=mdp.pedal_pos_b, params=priv_cmd),
    "foot_pos_ref_b": ObservationTermCfg(func=mdp.foot_pos_ref_b, params=priv_cmd),
    "pedal_pos_ref_b": ObservationTermCfg(func=mdp.pedal_pos_ref_b, params=priv_cmd),
    "foot_pedal_z_ref": ObservationTermCfg(func=mdp.foot_pedal_z_ref, params=priv_cmd),
    "foot_pedal_delta_b": ObservationTermCfg(
      func=mdp.foot_pedal_delta_b, params=priv_cmd
    ),
    "foot_pedal_delta_ref_b": ObservationTermCfg(
      func=mdp.foot_pedal_delta_ref_b, params=priv_cmd
    ),
    "foot_pos_err_b": ObservationTermCfg(func=mdp.foot_pos_err_b, params=priv_cmd),
    "pedal_pos_err_b": ObservationTermCfg(func=mdp.pedal_pos_err_b, params=priv_cmd),
    # contact force + bike joints (ref contact mask kept for rewards only, not obs)
    "bicycle_contact_force_mag": ObservationTermCfg(
      func=mdp.bicycle_contact_force_mag, params=priv_cmd
    ),
    "bicycle_joint_state": ObservationTermCfg(
      func=mdp.bicycle_joint_state, params=priv_cmd
    ),
    "bicycle_joint_state_ref": ObservationTermCfg(
      func=mdp.bicycle_joint_state_ref, params=priv_cmd
    ),
  }

  actor = dict(cfg.observations["actor"].terms)
  actor.update(privilege_terms)
  actor.pop("foot_pedal_lin_vel_b", None)
  actor.pop("foot_pedal_lin_vel_ref_b", None)
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor,
    concatenate_terms=cfg.observations["actor"].concatenate_terms,
    enable_corruption=cfg.observations["actor"].enable_corruption,
  )

  critic = dict(cfg.observations["critic"].terms)
  critic.update(privilege_terms)
  critic.pop("foot_pedal_lin_vel_b", None)
  critic.pop("foot_pedal_lin_vel_ref_b", None)
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic,
    concatenate_terms=cfg.observations["critic"].concatenate_terms,
    enable_corruption=cfg.observations["critic"].enable_corruption,
  )


# Proprio terms that get stacked history for Hist9 tasks (HDMI-style).
_PROPRIO_HISTORY_TERMS: tuple[str, ...] = (
  "joint_pos",
  "joint_vel",
  "actions",
  "base_ang_vel",
  "base_lin_vel",
  "motion_anchor_ori_b",
  "motion_anchor_pos_b",
)


def add_proprio_history(cfg, history_length: int = 9) -> None:
  """Stack the last ``history_length`` steps on proprio observation terms."""
  if history_length <= 0:
    return
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for key in _PROPRIO_HISTORY_TERMS:
      if key not in terms:
        continue
      terms[key].history_length = history_length
      terms[key].flatten_history_dim = True


def add_bike_interaction_local_points(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Append contact-aligned bike points + component ori (sim + ref).

  ``bike_interaction_points_*``: 15D handleL/R, saddle, pedalL/R contact pos.
  ``bike_component_ori_*``: 24D fork/frame/pedal_left/pedal_right ori6.
  Also keeps legacy mid-handle / saddle single-point terms.
  """
  priv_cmd: dict = {"command_name": "motion"}
  if saddle_local_pos is not None:
    priv_cmd["saddle_local_pos"] = saddle_local_pos
  saddle_params = dict(priv_cmd)
  ori_params = {"command_name": "motion"}
  extra: dict[str, ObservationTermCfg] = {
    "bike_handle_pos_b": ObservationTermCfg(
      func=mdp.bike_handle_pos_b, params={"command_name": "motion"}
    ),
    "bike_handle_pos_ref_b": ObservationTermCfg(
      func=mdp.bike_handle_pos_ref_b, params={"command_name": "motion"}
    ),
    "bike_saddle_pos_b": ObservationTermCfg(
      func=mdp.bike_saddle_pos_b, params=saddle_params
    ),
    "bike_saddle_pos_ref_b": ObservationTermCfg(
      func=mdp.bike_saddle_pos_ref_b, params=saddle_params
    ),
    "bike_interaction_points_b": ObservationTermCfg(
      func=mdp.bike_interaction_points_b, params=saddle_params
    ),
    "bike_interaction_points_ref_b": ObservationTermCfg(
      func=mdp.bike_interaction_points_ref_b, params=saddle_params
    ),
    "bike_component_ori_b": ObservationTermCfg(
      func=mdp.bike_component_ori_b, params=ori_params
    ),
    "bike_component_ori_ref_b": ObservationTermCfg(
      func=mdp.bike_component_ori_ref_b, params=ori_params
    ),
  }
  for group_name in ("actor", "critic"):
    terms = dict(cfg.observations[group_name].terms)
    terms.update(extra)
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=cfg.observations[group_name].concatenate_terms,
      enable_corruption=cfg.observations[group_name].enable_corruption,
    )


def add_rich_actor_pose_and_heights(cfg) -> None:
  """Add height / ankle lin-vel channels; drop robot body_pos/ori (FK⊂joint_pos).

  Adds / keeps:
    - motion_anchor_pos_b, base_lin_vel (normally stripped for no-state-est)
    - body_lin_vel / _ref = L/R ankle only (6D each), torso frame
    - key_body_height_w / _ref_w (pelvis, torso, L/R feet, L/R hands Z)

  Removes full-body lin/ang vel, foot_pedal_lin_vel, body_pos/ori, height_err.
  """
  priv_cmd = {"command_name": "motion"}
  critic_terms = cfg.observations["critic"].terms
  extra: dict[str, ObservationTermCfg] = {
    "body_lin_vel": ObservationTermCfg(
      func=mdp.ankle_lin_vel_b, params=priv_cmd
    ),
    "body_lin_vel_ref": ObservationTermCfg(
      func=mdp.ankle_lin_vel_ref_b, params=priv_cmd
    ),
    "key_body_height_w": ObservationTermCfg(
      func=mdp.key_body_height_w, params=priv_cmd
    ),
    "key_body_height_ref_w": ObservationTermCfg(
      func=mdp.key_body_height_ref_w, params=priv_cmd
    ),
  }
  if "motion_anchor_pos_b" in critic_terms:
    extra["motion_anchor_pos_b"] = critic_terms["motion_anchor_pos_b"]
  if "base_lin_vel" in critic_terms:
    extra["base_lin_vel"] = critic_terms["base_lin_vel"]

  for group_name in ("actor", "critic"):
    terms = dict(cfg.observations[group_name].terms)
    terms.pop("body_pos", None)
    terms.pop("body_ori", None)
    terms.pop("key_body_height_err", None)
    terms.pop("body_ang_vel", None)
    terms.pop("body_ang_vel_ref", None)
    terms.pop("foot_pedal_lin_vel_b", None)
    terms.pop("foot_pedal_lin_vel_ref_b", None)
    terms.update(extra)
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=cfg.observations[group_name].concatenate_terms,
      enable_corruption=cfg.observations[group_name].enable_corruption,
    )


def add_bicycle_rewards(
  cfg,
  *,
  curriculum: bool = False,
  handle_min_inner_plates: int = 1,
) -> None:
  """Append bike pose tracking + per-pair scheduled contact rewards.

  ``curriculum`` is kept for call-site compat but no longer changes contact
  weight / frc_thres (kinematic Curr-B unlock is separate).

  ``handle_min_inner_plates``: use 2 for ㄷ-hand (dhand), 1 for rubber capsule.
  """
  del curriculum
  cfg.rewards["bicycle_pos_tracking"] = RewardTermCfg(
    func=mdp.bicycle_pos_tracking_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.5},
  )
  cfg.rewards["bicycle_ori_tracking"] = RewardTermCfg(
    func=mdp.bicycle_ori_tracking_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.5},
  )
  cfg.rewards.pop("bicycle_eef_contact", None)
  add_split_bicycle_contact_rewards(
    cfg, handle_min_inner_plates=handle_min_inner_plates
  )
  # pedal_force / ground_push / forward_accel intentionally omitted.


def add_split_bicycle_contact_rewards(
  cfg,
  *,
  handle_min_inner_plates: int = 1,
) -> None:
  """Five schedule-gated contact terms, each smaller than body tracking.

  Group maxima are hands 0.5, saddle 0.5, and feet 1.0.  Before a valid
  contact, the distance half is active from 20 frames before scheduled onset
  and remains available for recovery.  Once valid contact is made, distance
  switches off and the bounded force half switches on.
  """
  shared = {
    "command_name": "motion",
    "pos_sigma": 0.3,
    "pos_tolerance": 0.02,
    "frc_sigma": 40.0,
    "frc_thres": 20.0,
    # Distance and force are mutually exclusive, so normalize either active
    # branch to [0, 1] and let the RewardTerm weight set the final pair cap.
    "approach_weight": 1.0,
    "force_weight": 1.0,
    "force_shaping": "tanh",
    "force_scale": 20.0,
    "precontact_frames": 20,
    "exclusive_distance_force": True,
    "pedal_z_margin": 0.0,
    "pedal_z_soft": 0.02,
    "handle_min_inner_plates": int(handle_min_inner_plates),
  }
  pair_term_weights = {
    "handle_left": 0.25,
    "handle_right": 0.25,
    "saddle": 0.5,
    "pedal_left": 0.5,
    "pedal_right": 0.5,
  }
  for key in BICYCLE_SPLIT_CONTACT_KEYS:
    cfg.rewards[BICYCLE_SPLIT_CONTACT_REWARD_NAMES[key]] = RewardTermCfg(
      func=mdp.bicycle_contact_pair_exp,
      weight=pair_term_weights[key],
      params={**shared, "pair_key": key},
    )


def _update_split_contact_params(cfg, **updates) -> None:
  for name in BICYCLE_SPLIT_CONTACT_REWARD_NAMES.values():
    term = cfg.rewards.get(name)
    if term is not None:
      term.params.update(updates)
  legacy = cfg.rewards.get("bicycle_eef_contact")
  if legacy is not None:
    legacy.params.update(updates)


def add_bicycle_curriculum(
  cfg,
  mode: Literal["A", "B", "C", "bike052", "PD"],
  *,
  decay_until_iter: float | None = None,
  pelvis_push_target_speed: float = 1.0,
  pelvis_push_start_frame: int = 550,
  pelvis_push_end_frame: int = 750,
  saddle_contact_local: tuple[float, float, float] | None = None,
  pd_assist_levels: tuple[float, ...] = (0.0, 0.05, 0.2, 0.5, 1.0),
) -> None:
  """Dynamic bike curriculum: virtual wrench (A), partial lock (B), C, or bike052.

  C = no kin lock; decaying roll anti-fall + timed forward push (230–280).
  bike052 = freewheel pedals + 30k decaying roll hold + pelvis world-+X push (11–15 s).
  """
  if mode == "A":
    step_func = mdp.apply_bicycle_curriculum_step_a
  elif mode == "B":
    step_func = mdp.apply_bicycle_curriculum_step_b
  elif mode == "C":
    step_func = mdp.apply_bicycle_curriculum_step_c
  elif mode == "bike052":
    step_func = mdp.apply_bicycle_curriculum_step_bike052
  elif mode == "PD":
    step_func = mdp.apply_bicycle_physical_pd_assistance
  else:
    raise ValueError(f"Unknown bicycle curriculum mode: {mode}")
  init_params: dict = {"mode": mode}
  if mode in ("bike052", "PD"):
    init_params.update(
      {
        "decay_until_iter": decay_until_iter
        if decay_until_iter is not None
        else (30_000 if mode == "bike052" else 50_000),
        "pelvis_push_target_speed": pelvis_push_target_speed,
        "pelvis_push_start_frame": pelvis_push_start_frame,
        "pelvis_push_end_frame": pelvis_push_end_frame,
        "saddle_contact_local": saddle_contact_local,
      }
    )
  if mode == "PD":
    init_params["pd_assist_levels"] = pd_assist_levels
  cfg.events["bicycle_curriculum_init"] = EventTermCfg(
    func=mdp.init_bicycle_curriculum_state,
    mode="startup",
    params=init_params,
  )
  cfg.events["bicycle_curriculum_step"] = EventTermCfg(
    func=step_func,
    mode="step",
    params={"command_name": "motion"},
  )
  cfg.curriculum = {
    "bicycle_contact": CurriculumTermCfg(
      # Logs lock scales only; does not mutate EE contact reward weights.
      func=mdp.update_bicycle_curriculum,
      params={"reward_name": "bicycle_eef_contact"},
    ),
  }


def configure_bike052_rewards(
  cfg,
  *,
  handle_min_inner_plates: int = 1,
) -> None:
  """Normalize bike052 rewards: static EE contact, drop force-push extras."""
  for name in (
    "bicycle_pedal_force",
    "bicycle_left_foot_ground_push",
    "bicycle_forward_accel_push",
  ):
    cfg.rewards.pop(name, None)
  if any(n in cfg.rewards for n in BICYCLE_SPLIT_CONTACT_REWARD_NAMES.values()):
    _update_split_contact_params(
      cfg,
      frc_thres=20.0,
      handle_min_inner_plates=int(handle_min_inner_plates),
    )
  elif "bicycle_eef_contact" in cfg.rewards:
    term = cfg.rewards["bicycle_eef_contact"]
    term.func = mdp.bicycle_eef_contact_exp
    term.weight = 1.0
    term.params.update(
      {
        "frc_thres": 20.0,
        "handle_min_inner_plates": int(handle_min_inner_plates),
        "pair_weights": dict(BICYCLE_EEF_CONTACT_PAIR_WEIGHTS),
      }
    )


_SLIM_BIKE_OBS_KEEP: tuple[str, ...] = (
  "command",
  "command_future",
  "joint_pos",
  "joint_vel",
  "joint_torque",
  "actions",
  "base_ang_vel",
  "base_lin_vel",
  "base_lin_vel_ref",
  "motion_anchor_ori_b",
  "motion_anchor_pos_b",
  "key_body_height_w",
  "key_body_height_ref_w",
  "bike_interaction_points_b",
  "bike_component_ori_b",
  "bicycle_joint_state",
  "bicycle_joint_state_ref",
)

# Legacy mid-handle pose pack (36D). Prefer contact points + ori below.
_RESMIMIC_RESIDUAL_ACTOR_KEEP: tuple[str, ...] = (
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
  "base_ang_vel",
  "base_lin_vel",
  "motion_anchor_ori_b",
  "motion_anchor_pos_b",
  "bike_component_pose_b",
)


def configure_resmimic_residual_actor_obs(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Legacy slim actor: Hist proprio + ``bike_component_pose_b`` (36D).

  New experiments should use
  ``configure_resmimic_residual_actor_obs_contact`` instead.
  """
  params: dict = {"command_name": "motion"}
  if saddle_local_pos is not None:
    params["saddle_local_pos"] = saddle_local_pos

  actor_terms = dict(cfg.observations["actor"].terms)
  actor_terms["bike_component_pose_b"] = ObservationTermCfg(
    func=mdp.bike_component_pose_b, params=params
  )
  slim = {
    k: actor_terms[k]
    for k in _RESMIMIC_RESIDUAL_ACTOR_KEEP
    if k in actor_terms
  }
  if "bike_component_pose_b" not in slim:
    slim["bike_component_pose_b"] = actor_terms["bike_component_pose_b"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=slim,
    concatenate_terms=cfg.observations["actor"].concatenate_terms,
    enable_corruption=False,
  )

  critic_terms = dict(cfg.observations["critic"].terms)
  critic_terms["bike_component_pose_b"] = ObservationTermCfg(
    func=mdp.bike_component_pose_b, params=params
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=cfg.observations["critic"].concatenate_terms,
    enable_corruption=cfg.observations["critic"].enable_corruption,
  )


def configure_slim_bike_obs(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
  include_ori: bool = True,
) -> None:
  """Actor = critic slim pack for new bike experiments.

  Keeps proprio/hist, heights, ref root lin-vel, contact points, component ori,
  bike joint state, and ``command_future`` (+1..+5 ref joint frames).
  Drops ankle body lin-vel (implicit from joint/root) and legacy privilege.
  """
  params: dict = {"command_name": "motion"}
  if saddle_local_pos is not None:
    params["saddle_local_pos"] = saddle_local_pos
  cmd = {"command_name": "motion"}

  terms: dict[str, ObservationTermCfg] = {
    "command_future": ObservationTermCfg(
      func=mdp.motion_command_future, params=cmd
    ),
    "joint_torque": ObservationTermCfg(func=mdp.joint_torque),
    "base_lin_vel_ref": ObservationTermCfg(
      func=mdp.base_lin_vel_ref, params=cmd
    ),
    "key_body_height_w": ObservationTermCfg(
      func=mdp.key_body_height_w, params=cmd
    ),
    "key_body_height_ref_w": ObservationTermCfg(
      func=mdp.key_body_height_ref_w, params=cmd
    ),
    "bike_interaction_points_b": ObservationTermCfg(
      func=mdp.bike_interaction_points_b, params=params
    ),
    "bicycle_joint_state": ObservationTermCfg(
      func=mdp.bicycle_joint_state, params=cmd
    ),
    "bicycle_joint_state_ref": ObservationTermCfg(
      func=mdp.bicycle_joint_state_ref, params=cmd
    ),
  }
  if include_ori:
    terms["bike_component_ori_b"] = ObservationTermCfg(
      func=mdp.bike_component_ori_b, params=cmd
    )

  keep = list(_SLIM_BIKE_OBS_KEEP)
  if not include_ori:
    keep = [k for k in keep if k != "bike_component_ori_b"]

  for group_name in ("actor", "critic"):
    merged = dict(cfg.observations[group_name].terms)
    merged.update(terms)
    # Ensure base proprio terms exist from tracking cfg.
    slim = {k: merged[k] for k in keep if k in merged}
    for k in keep:
      if k in terms:
        slim[k] = terms[k]
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=slim,
      concatenate_terms=cfg.observations[group_name].concatenate_terms,
      enable_corruption=(
        False
        if group_name == "actor"
        else cfg.observations[group_name].enable_corruption
      ),
    )


def configure_deployable_hist5_obs(
  cfg,
  *,
  include_bicycle_state: bool = False,
  include_future_reference: bool = False,
  include_interaction_geometry: bool = False,
  saddle_local_pos: tuple[float, float, float] | None = None,
  history_length: int = 5,
) -> None:
  """Replace actor/critic observations with a deployable Hist5 pack.

  Base pack (528-D at Hist5): current reference robot q/qdot, robot joint
  q/qdot, torso IMU angular velocity and projected gravity, previous actions,
  and the five reference interaction-contact flags.  The proprioceptive/IMU/
  action channels use history; command and contact intent do not.

  ``include_future_reference`` adds five +1..+5 policy-step reference robot
  q/qdot frames (5 x 58 = 290-D). ``include_bicycle_state`` adds an 8-D current-bike pack measurable using a
  bicycle IMU plus steer/crank/rear-wheel encoders.  Actor and critic receive
  the same terms so this is a strict deployable-observation ablation.

  ``include_interaction_geometry`` additionally exposes the current simulated
  bicycle's five interaction points (15-D) and four component orientations
  (24-D), all expressed in the current robot-anchor frame.
  """
  if history_length <= 0:
    raise ValueError("history_length must be positive")

  def _history_term(func, *, params: dict | None = None) -> ObservationTermCfg:
    return ObservationTermCfg(
      func=func,
      params={} if params is None else params,
      history_length=history_length,
      flatten_history_dim=True,
    )

  def _terms() -> dict[str, ObservationTermCfg]:
    cmd = {"command_name": "motion"}
    terms: dict[str, ObservationTermCfg] = {
      # Current reference robot joint position + velocity (29 + 29).
      "command": ObservationTermCfg(func=mdp.generated_commands, params=cmd),
      "joint_pos": _history_term(mdp.joint_pos_rel),
      "joint_vel": _history_term(mdp.joint_vel_rel),
      "base_ang_vel": _history_term(
        mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
      ),
      "projected_gravity": _history_term(mdp.projected_gravity),
      "actions": _history_term(mdp.last_action),
      "bicycle_ref_contact_mask": ObservationTermCfg(
        func=mdp.bicycle_ref_contact_mask, params=cmd
      ),
    }
    if include_future_reference:
      terms["command_future"] = ObservationTermCfg(
        func=mdp.motion_command_future,
        params={**cmd, "future_steps": (1, 2, 3, 4, 5)},
      )
    if include_bicycle_state:
      terms["bicycle_deployable_state"] = ObservationTermCfg(
        func=mdp.bicycle_deployable_state, params=cmd
      )
    if include_interaction_geometry:
      interaction_params = dict(cmd)
      if saddle_local_pos is not None:
        interaction_params["saddle_local_pos"] = saddle_local_pos
      terms["bike_interaction_points_b"] = ObservationTermCfg(
        func=mdp.bike_interaction_points_b, params=interaction_params
      )
      terms["bike_component_ori_b"] = ObservationTermCfg(
        func=mdp.bike_component_ori_b, params=cmd
      )
    return terms

  for group_name in ("actor", "critic"):
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=_terms(), concatenate_terms=True, enable_corruption=False
    )


def configure_bicycle_distill_observations(
  cfg,
  *,
  history_length: int = 5,
) -> None:
  """Create asymmetric observations for privileged-teacher distillation.

  The existing privileged PhysicsPD actor observation is preserved verbatim as
  the ``teacher`` group.  The student ``actor`` gets only the deployable Hist5
  pack, five-step reference lookahead, and reference phase/contact intent. No
  bicycle state is exposed because the target hardware has no bicycle-mounted
  sensors. The original privileged ``critic`` group is deliberately left
  unchanged for asymmetric PPO.

  Keeping these as three separate manager groups is important: the frozen
  teacher and student are evaluated at the same simulator state, while the
  exported student policy consumes only ``actor``.
  """
  teacher_group = cfg.observations["actor"]
  cfg.observations["teacher"] = ObservationGroupCfg(
    terms=dict(teacher_group.terms),
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Reuse the deployment contract already exercised by the kinematic
  # ablations. This helper replaces actor and critic, so restore the privileged
  # critic immediately afterwards.
  privileged_critic = cfg.observations["critic"]
  configure_deployable_hist5_obs(
    cfg,
    include_bicycle_state=False,
    include_future_reference=True,
    include_interaction_geometry=False,
    history_length=history_length,
  )
  actor_terms = dict(cfg.observations["actor"].terms)
  actor_terms["reference_motion_phase"] = ObservationTermCfg(
    func=mdp.reference_motion_phase,
    params={"command_name": "motion"},
  )
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["critic"] = privileged_critic


def configure_bicycle_dual_privileged_latent_observations(cfg) -> None:
  """Split the V3 actor input for an RMA-style dual-latent teacher.

  Known reference intent and deployable proprioception remain in ``actor``.
  Simulator-only robot and bicycle state are routed through separate groups so
  the custom actor can compress each group to an independent 8-D latent.  The
  original critic group is intentionally preserved verbatim.
  """
  full_terms = dict(cfg.observations["actor"].terms)
  regular_names = (
    "command",
    "command_future",
    "joint_pos",
    "joint_vel",
    "actions",
    "base_ang_vel",
    "base_lin_vel_ref",
    "key_body_height_ref_w",
    "bicycle_joint_state_ref",
    "bicycle_root_velocity_ref_b",
    "reference_motion_phase",
    "bicycle_ref_contact_mask",
    "bicycle_contact_time_to_transition",
    "bicycle_pd_assistance_scale",
  )
  robot_privileged_names = (
    "joint_torque",
    "base_lin_vel",
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "key_body_height_w",
  )
  bicycle_privileged_names = (
    "bike_interaction_points_b",
    "bike_component_ori_b",
    "bicycle_joint_state",
    "bicycle_crank_phase",
    "bicycle_contact_force_mag",
    "bicycle_root_velocity_b",
  )
  assigned = set(regular_names + robot_privileged_names + bicycle_privileged_names)
  missing = assigned.difference(full_terms)
  unassigned = set(full_terms).difference(assigned)
  if missing or unassigned:
    raise KeyError(
      "Dual-latent observation partition is stale: "
      f"missing={sorted(missing)}, unassigned={sorted(unassigned)}"
    )

  def _group(names: tuple[str, ...]) -> ObservationGroupCfg:
    return ObservationGroupCfg(
      terms={name: full_terms[name] for name in names},
      concatenate_terms=True,
      enable_corruption=False,
    )

  cfg.observations["actor"] = _group(regular_names)
  cfg.observations["robot_privileged"] = _group(robot_privileged_names)
  cfg.observations["bicycle_privileged"] = _group(bicycle_privileged_names)


def configure_resmimic_residual_actor_obs_contact(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
  include_ori: bool = True,
  include_ref: bool = False,
) -> None:
  """Slim actor=critic for residual experiments (contact points + ori + future)."""
  del include_ref  # bike ref points omitted; command_future carries ref lookahead
  configure_slim_bike_obs(
    cfg, saddle_local_pos=saddle_local_pos, include_ori=include_ori
  )


def configure_physics_pd_actor_obs(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float],
  include_acceleration: bool = False,
) -> None:
  """Add phase, contact intent, and Markov bicycle dynamics to the slim pack."""
  configure_slim_bike_obs(
    cfg, saddle_local_pos=saddle_local_pos, include_ori=True
  )
  cmd = {"command_name": "motion"}
  additions: dict[str, ObservationTermCfg] = {
    "reference_motion_phase": ObservationTermCfg(
      func=mdp.reference_motion_phase, params=cmd
    ),
    "bicycle_crank_phase": ObservationTermCfg(
      func=mdp.bicycle_crank_phase, params=cmd
    ),
    "bicycle_ref_contact_mask": ObservationTermCfg(
      func=mdp.bicycle_ref_contact_mask, params=cmd
    ),
    "bicycle_contact_time_to_transition": ObservationTermCfg(
      func=mdp.bicycle_contact_time_to_transition,
      params={**cmd, "max_horizon_s": 5.0},
    ),
    "bicycle_contact_force_mag": ObservationTermCfg(
      func=mdp.bicycle_contact_force_mag, params=cmd
    ),
    "bicycle_root_velocity_b": ObservationTermCfg(
      func=mdp.bicycle_root_velocity_b, params=cmd
    ),
    "bicycle_root_velocity_ref_b": ObservationTermCfg(
      func=mdp.bicycle_root_velocity_ref_b, params=cmd
    ),
    "bicycle_pd_assistance_scale": ObservationTermCfg(
      func=mdp.bicycle_pd_assistance_scale, params=cmd
    ),
    # Multi-horizon intent: 0.02, 0.1, 0.2, 0.4, 0.8 s at 50 Hz.
    "command_future": ObservationTermCfg(
      func=mdp.motion_command_future,
      params={**cmd, "future_steps": (1, 5, 10, 20, 40)},
    ),
  }
  if include_acceleration:
    additions["bicycle_root_acceleration_b"] = ObservationTermCfg(
      func=mdp.bicycle_root_acceleration_b, params=cmd
    )
  for group_name in ("actor", "critic"):
    terms = dict(cfg.observations[group_name].terms)
    terms.update(additions)
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=False,
    )


def configure_physics_pd_rewards(cfg) -> None:
  """Replace ghost-pose objectives with physical riding objectives."""
  cfg.rewards.pop("bicycle_pos_tracking", None)
  cfg.rewards.pop("bicycle_ori_tracking", None)
  _update_split_contact_params(
    cfg,
    position_target="simulated",
  )
  cmd = {"command_name": "motion"}
  cfg.rewards["bicycle_roll_dynamics"] = RewardTermCfg(
    func=mdp.bicycle_roll_dynamics_tracking_exp,
    weight=2.0,
    params={**cmd, "lean_std": 0.2, "roll_rate_std": 1.0},
  )
  cfg.rewards["bicycle_forward_speed"] = RewardTermCfg(
    func=mdp.bicycle_forward_speed_tracking_exp,
    weight=1.5,
    params={**cmd, "std": 0.5},
  )
  cfg.rewards["bicycle_crank_cadence"] = RewardTermCfg(
    func=mdp.bicycle_crank_cadence_tracking_exp,
    weight=1.0,
    params={**cmd, "std": 2.0},
  )
  cfg.rewards["bicycle_rear_wheel_no_slip"] = RewardTermCfg(
    func=mdp.bicycle_rear_wheel_no_slip_exp,
    weight=0.5,
    params={**cmd, "wheel_radius": 0.24, "std": 0.5},
  )


def configure_crank_progress_top_pedal_rewards(cfg) -> None:
  """Crank progress + sole-pad-to-pedal contact.

  The historical function/task name contains ``top_pedal``, but a bicycle
  pedal rotates.  Reward any physical contact between a thin robot sole layer
  and its pedal; distance is measured from the midfoot sole point.  This still
  rejects ankle, foot-top, and main-foot-box contacts because the sensor's
  primary geom match is sole-only.
  """
  _update_split_contact_params(
    cfg,
    pedal_top_contact=False,
    pedal_sole_contact=True,
  )

  # Physical tire friction already enforces rolling.  Keep its error as a metric
  # instead of granting an almost-constant positive reward.
  cfg.rewards.pop("bicycle_rear_wheel_no_slip", None)
  cmd = {"command_name": "motion"}
  cfg.rewards["bicycle_crank_forward_progress"] = RewardTermCfg(
    func=mdp.BicycleCrankWindowProgress,
    weight=2.0,
    params={
      **cmd,
      "window_steps": 10,
      "forward_sign": 1.0,
      "min_target_rate": 0.5,
    },
  )
  cfg.rewards["bicycle_crank_backward"] = RewardTermCfg(
    func=mdp.bicycle_crank_backward_rate_penalty,
    weight=-2.5,
    params={
      **cmd,
      "forward_sign": 1.0,
      "tolerance": 0.15,
      "rate_scale": 2.0,
    },
  )
  cfg.rewards["bicycle_crank_reversal"] = RewardTermCfg(
    func=mdp.BicycleCrankDirectionReversal,
    weight=-0.5,
    params={**cmd, "forward_sign": 1.0, "min_abs_rate": 0.25},
  )
  # Do not classify pedal faces: the platform rotates, and any contact made by
  # the robot's sole collision layer is a valid foot-pedal contact.
  cfg.rewards.pop("bicycle_pedal_underside_contact", None)

  if cfg.metrics is None:
    cfg.metrics = {}
  cfg.metrics["rear_wheel_longitudinal_slip_abs"] = MetricsTermCfg(
    func=mdp.bicycle_rear_wheel_longitudinal_slip_abs,
    params={**cmd, "wheel_radius": 0.24},
  )


def configure_contact_30_20_50_no_backward(cfg) -> None:
  """Bias interaction learning toward pedals without blocking exploration.

  Contact is five independent schedule-gated terms, so a missing pedal can no
  longer be hidden inside a handle/saddle average.  Group 30/20/50 averaging
  is obsolete.  Remove the instantaneous backward-rate cost while retaining
  windowed forward progress, direction-reversal, and sole-pad contact terms.
  """
  cfg.rewards.pop("bicycle_crank_backward", None)


def configure_ratchet_ablation_rewards(
  cfg,
  *,
  backward_weight: float | None,
  lower_body_half: bool,
  pedal_miss_termination: bool,
  pedal_distance_termination: bool = False,
) -> None:
  """Restore the successful 2026-08-17 reward and add ratchet ablations.

  Forbidden robot-bike contact and post-schedule handle detach are common to
  every arm. PedalMiss10 termination, instantaneous backward-rate weight, and
  half-strength lower-body tracking are the controlled comparison axes.
  """
  for reward_name in BICYCLE_SPLIT_CONTACT_REWARD_NAMES.values():
    cfg.rewards.pop(reward_name, None)
  cfg.rewards["bicycle_eef_contact"] = RewardTermCfg(
    func=mdp.bicycle_eef_contact_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "pos_sigma": 0.3,
      "pos_tolerance": 0.02,
      "frc_sigma": 40.0,
      "frc_thres": 20.0,
      "gain": 5.0,
      "pedal_z_margin": 0.0,
      "pedal_z_soft": 0.02,
      "handle_min_inner_plates": 2,
      "pair_weights": {
        "handle_left": 15.0,
        "handle_right": 15.0,
        "saddle": 20.0,
        "pedal_left": 25.0,
        "pedal_right": 25.0,
        "foot_left_ground": 0.0,
        "foot_right_ground": 0.0,
      },
      "position_target": "simulated",
      "require_contact": True,
      "pedal_top_contact": False,
      "pedal_sole_contact": True,
      # Exact successful-run semantics: all foot1--7 sole contacts count.
      "pedal_forefoot_filter": False,
    },
  )

  if backward_weight is None:
    cfg.rewards.pop("bicycle_crank_backward", None)
  else:
    cfg.rewards["bicycle_crank_backward"] = RewardTermCfg(
      func=mdp.bicycle_crank_backward_rate_penalty,
      weight=-abs(float(backward_weight)),
      params={
        "command_name": "motion",
        "forward_sign": 1.0,
        "tolerance": 0.15,
        "rate_scale": 2.0,
      },
    )

  cfg.rewards["forbidden_robot_bicycle_collision"] = RewardTermCfg(
    func=mdp.contact_force_history_violation_cost,
    weight=-1.0,
    params={
      "sensor_name": FORBIDDEN_BIKE_CONTACT_SENSOR,
      "force_threshold": 10.0,
      "max_count": 1,
    },
  )
  for pair_key in ("handle_left", "handle_right"):
    cfg.rewards[f"post_schedule_{pair_key}_contact"] = RewardTermCfg(
      func=mdp.bicycle_post_schedule_contact,
      weight=-0.5,
      params={
        "command_name": "motion",
        "pair_key": pair_key,
        "sensor_name": HANDLE_DETACH_SENSOR_NAMES[pair_key],
        "grace_frames": 5,
      },
    )

  if pedal_miss_termination:
    cfg.terminations["pedal_contact_missing_10"] = TerminationTermCfg(
      func=mdp.BicyclePedalContactMissingTermination,
      params={
        "command_name": "motion",
        "left_sensor_name": "contact_pedal_left",
        "right_sensor_name": "contact_pedal_right",
        "max_missing_steps": 10,
        "force_threshold": 0.1,
      },
      time_out=False,
    )
  else:
    cfg.terminations.pop("pedal_contact_missing_10", None)

  if pedal_distance_termination:
    cfg.terminations["pedal_distance_over_10cm_10"] = TerminationTermCfg(
      func=mdp.BicyclePedalDistanceMissingTermination,
      params={
        "command_name": "motion",
        "max_distance": 0.10,
        "max_far_steps": 10,
      },
      time_out=False,
    )
  else:
    cfg.terminations.pop("pedal_distance_over_10cm_10", None)

  if lower_body_half:
    upper_bodies = (
      "pelvis",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    )
    tracking_terms = {
      "motion_body_pos": (mdp.motion_relative_body_position_error_exp, 0.3),
      "motion_body_ori": (mdp.motion_relative_body_orientation_error_exp, 0.4),
      "motion_body_lin_vel": (mdp.motion_global_body_linear_velocity_error_exp, 1.0),
      "motion_body_ang_vel": (mdp.motion_global_body_angular_velocity_error_exp, 3.14),
    }
    for name, (func, std) in tracking_terms.items():
      # 0.5*all + 0.5*upper preserves a maximum of 1 while halving the
      # influence of the six hip-roll/knee/ankle reference bodies.
      cfg.rewards[name].weight = 0.5
      cfg.rewards[f"{name}_upper"] = RewardTermCfg(
        func=func,
        weight=0.5,
        params={"command_name": "motion", "std": std, "body_names": upper_bodies},
      )


def configure_deployable_lesson_rewards(cfg, *, fixed_roll_pd: bool) -> None:
  """Keep only policy-controllable bicycle rewards for the staged lessons.

  A kinematically replayed bicycle makes roll, speed, crank cadence/progress,
  and reversals independent of the robot action.  Those terms therefore act as
  phase-dependent constants and are removed; physical EEF contact remains
  because the robot still controls whether its hands, pelvis, and soles touch.

  With fixed roll PD the bicycle and freewheel are dynamic, but this lesson is
  deliberately restricted to pedal interaction.  Keep cadence, windowed
  forward crank progress, and reversal together with EEF contact; remove roll
  tracking (handled by the assistant) and forward-speed tracking (reserved for
  a later propulsion lesson).
  """
  remove = {
    "bicycle_roll_dynamics",
    "bicycle_forward_speed",
  }
  if not fixed_roll_pd:
    remove.update(
      {
        "bicycle_crank_cadence",
        "bicycle_crank_forward_progress",
        "bicycle_crank_reversal",
      }
    )
  for name in remove:
    cfg.rewards.pop(name, None)


def configure_resmimic_residual_actor_obs_pos_only(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Legacy: proprio + mid-handle ``bike_component_pos_b`` (12D).

  Prefer ``configure_resmimic_residual_actor_obs_contact(..., include_ori=False)``.
  """
  configure_resmimic_residual_actor_obs_contact(
    cfg, saddle_local_pos=saddle_local_pos, include_ori=False, include_ref=False
  )


def configure_pedal_window_slim_rewards(cfg) -> None:
  """Keep robot tracking + eef contact; drop bike pose / pedal-force extras."""
  for name in (
    "bicycle_pos_tracking",
    "bicycle_ori_tracking",
    "bicycle_pedal_force",
    "bicycle_left_foot_ground_push",
    "bicycle_forward_accel_push",
  ):
    cfg.rewards.pop(name, None)


def add_pelvis_saddle_lock_curriculum(
  cfg,
  *,
  decay_until_iter: float = 30000.0,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Startup + per-step pelvis→saddle lock with linear unlock."""
  params: dict = {
    "command_name": "motion",
    "decay_until_iter": float(decay_until_iter),
  }
  if saddle_local_pos is not None:
    params["saddle_local_pos"] = saddle_local_pos
  cfg.events["pelvis_saddle_lock_init"] = EventTermCfg(
    func=mdp.init_pelvis_saddle_lock,
    mode="startup",
    params=params,
  )
  cfg.events["pelvis_saddle_lock_step"] = EventTermCfg(
    func=mdp.apply_pelvis_saddle_lock_step,
    mode="step",
    params={"command_name": "motion"},
  )
  cfg.curriculum["pelvis_saddle_lock"] = CurriculumTermCfg(
    func=mdp.update_pelvis_saddle_lock_curriculum,
  )


def configure_legacy_hist9_rich_1490_obs(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Restore Aug-9 Hist9-Rich actor=critic obs (1490D) for old checkpoint play.

  Term order matches train logs for
  ``…Curriculum-B-Hist9-Rich-Bike052-Dhand``.
  """
  cmd = {"command_name": "motion"}
  params = dict(cmd)
  if saddle_local_pos is not None:
    params["saddle_local_pos"] = saddle_local_pos

  # Exact actor term order from 1490D training runs.
  ordered: list[tuple[str, ObservationTermCfg]] = [
    ("command", ObservationTermCfg(func=mdp.generated_commands, params=cmd)),
    ("motion_anchor_pos_b", ObservationTermCfg(func=mdp.motion_anchor_pos_b, params=cmd)),
    ("motion_anchor_ori_b", ObservationTermCfg(func=mdp.motion_anchor_ori_b, params=cmd)),
    ("base_lin_vel", ObservationTermCfg(func=mdp.base_lin_vel)),
    ("base_ang_vel", ObservationTermCfg(func=mdp.base_ang_vel)),
    ("joint_pos", ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True})),
    ("joint_vel", ObservationTermCfg(func=mdp.joint_vel_rel)),
    ("actions", ObservationTermCfg(func=mdp.last_action)),
    ("bicycle_ref_points_b", ObservationTermCfg(func=mdp.bicycle_ref_points_b, params=cmd)),
    ("bicycle_sim_points_b", ObservationTermCfg(func=mdp.bicycle_sim_points_b, params=cmd)),
    ("bicycle_ref_ori_b", ObservationTermCfg(func=mdp.bicycle_ref_ori_b, params=cmd)),
    ("bicycle_sim_ori_b", ObservationTermCfg(func=mdp.bicycle_sim_ori_b, params=cmd)),
    ("bicycle_ref_lin_vel_b", ObservationTermCfg(func=mdp.bicycle_ref_lin_vel_b, params=cmd)),
    ("bicycle_sim_lin_vel_b", ObservationTermCfg(func=mdp.bicycle_sim_lin_vel_b, params=cmd)),
    ("foot_pos_b", ObservationTermCfg(func=mdp.foot_pos_b, params=cmd)),
    ("pedal_pos_b", ObservationTermCfg(func=mdp.pedal_pos_b, params=cmd)),
    ("foot_pos_ref_b", ObservationTermCfg(func=mdp.foot_pos_ref_b, params=cmd)),
    ("pedal_pos_ref_b", ObservationTermCfg(func=mdp.pedal_pos_ref_b, params=cmd)),
    ("foot_pedal_z_ref", ObservationTermCfg(func=mdp.foot_pedal_z_ref, params=cmd)),
    ("foot_pedal_delta_b", ObservationTermCfg(func=mdp.foot_pedal_delta_b, params=cmd)),
    ("foot_pedal_delta_ref_b", ObservationTermCfg(func=mdp.foot_pedal_delta_ref_b, params=cmd)),
    ("foot_pedal_lin_vel_b", ObservationTermCfg(func=mdp.foot_pedal_lin_vel_b, params=cmd)),
    ("foot_pedal_lin_vel_ref_b", ObservationTermCfg(func=mdp.foot_pedal_lin_vel_ref_b, params=cmd)),
    ("foot_pos_err_b", ObservationTermCfg(func=mdp.foot_pos_err_b, params=cmd)),
    ("pedal_pos_err_b", ObservationTermCfg(func=mdp.pedal_pos_err_b, params=cmd)),
    ("bicycle_ref_contact_mask", ObservationTermCfg(func=mdp.bicycle_ref_contact_mask, params=cmd)),
    ("bicycle_contact_force_mag", ObservationTermCfg(func=mdp.bicycle_contact_force_mag, params=cmd)),
    ("bicycle_joint_state", ObservationTermCfg(func=mdp.bicycle_joint_state_legacy4, params=cmd)),
    ("bicycle_joint_state_ref", ObservationTermCfg(func=mdp.bicycle_joint_state_ref_legacy4, params=cmd)),
    ("bike_handle_pos_b", ObservationTermCfg(func=mdp.bike_handle_pos_b, params=cmd)),
    ("bike_handle_pos_ref_b", ObservationTermCfg(func=mdp.bike_handle_pos_ref_b, params=cmd)),
    ("bike_saddle_pos_b", ObservationTermCfg(func=mdp.bike_saddle_pos_b, params=params)),
    ("bike_saddle_pos_ref_b", ObservationTermCfg(func=mdp.bike_saddle_pos_ref_b, params=params)),
    ("bike_interaction_points_b", ObservationTermCfg(func=mdp.bike_interaction_points_b_legacy12, params=params)),
    ("bike_interaction_points_ref_b", ObservationTermCfg(func=mdp.bike_interaction_points_ref_b_legacy12, params=params)),
    ("body_pos", ObservationTermCfg(func=mdp.robot_body_pos_b, params=cmd)),
    ("body_ori", ObservationTermCfg(func=mdp.robot_body_ori_b, params=cmd)),
    ("body_lin_vel", ObservationTermCfg(func=mdp.robot_body_lin_vel_b, params=cmd)),
    ("body_lin_vel_ref", ObservationTermCfg(func=mdp.robot_body_lin_vel_ref_b, params=cmd)),
    ("body_ang_vel", ObservationTermCfg(func=mdp.robot_body_ang_vel_b, params=cmd)),
    ("body_ang_vel_ref", ObservationTermCfg(func=mdp.robot_body_ang_vel_ref_b, params=cmd)),
    ("key_body_height_w", ObservationTermCfg(func=mdp.key_body_height_w_legacy4, params=cmd)),
    ("key_body_height_ref_w", ObservationTermCfg(func=mdp.key_body_height_ref_w_legacy4, params=cmd)),
    ("key_body_height_err", ObservationTermCfg(func=mdp.key_body_height_err_legacy4, params=cmd)),
  ]
  terms = dict(ordered)
  # Preserve hist9 on proprio channels if already set on the env.
  base_actor = cfg.observations["actor"].terms
  for key in _PROPRIO_HISTORY_TERMS:
    if key in terms and key in base_actor:
      terms[key].history_length = getattr(base_actor[key], "history_length", None) or 9
      terms[key].flatten_history_dim = True
    elif key in terms:
      terms[key].history_length = 9
      terms[key].flatten_history_dim = True

  for group_name in ("actor", "critic"):
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=dict(terms),
      concatenate_terms=True,
      enable_corruption=False,
    )


def configure_legacy_resmimic_988_obs(
  cfg,
  *,
  saddle_local_pos: tuple[float, float, float] | None = None,
) -> None:
  """Restore Aug-10 ResMimic residual actor 988D / critic 1466D for play."""
  cmd = {"command_name": "motion"}
  params = dict(cmd)
  if saddle_local_pos is not None:
    params["saddle_local_pos"] = saddle_local_pos

  actor_keys = (
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "base_ang_vel",
    "base_lin_vel",
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "bike_component_pos_b",
  )
  merged = dict(cfg.observations["actor"].terms)
  merged.update(dict(cfg.observations["critic"].terms))
  merged["bike_component_pos_b"] = ObservationTermCfg(
    func=mdp.bike_component_pos_b, params=params
  )
  # Ensure proprio hist9.
  for key in _PROPRIO_HISTORY_TERMS:
    if key in merged:
      merged[key].history_length = 9
      merged[key].flatten_history_dim = True

  actor = {k: merged[k] for k in actor_keys if k in merged}
  actor["bike_component_pos_b"] = merged["bike_component_pos_b"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor,
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Critic: privileged pack + heights/vels + bike_component_pos (1466D log).
  critic_keys = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "bicycle_ref_points_b",
    "bicycle_sim_points_b",
    "bicycle_ref_ori_b",
    "bicycle_sim_ori_b",
    "bicycle_ref_lin_vel_b",
    "bicycle_sim_lin_vel_b",
    "foot_pos_b",
    "pedal_pos_b",
    "foot_pos_ref_b",
    "pedal_pos_ref_b",
    "foot_pedal_z_ref",
    "foot_pedal_delta_b",
    "foot_pedal_delta_ref_b",
    "foot_pedal_lin_vel_b",
    "foot_pedal_lin_vel_ref_b",
    "foot_pos_err_b",
    "pedal_pos_err_b",
    "bicycle_ref_contact_mask",
    "bicycle_contact_force_mag",
    "bicycle_joint_state",
    "bicycle_joint_state_ref",
    "body_lin_vel",
    "body_lin_vel_ref",
    "body_ang_vel",
    "body_ang_vel_ref",
    "key_body_height_w",
    "key_body_height_ref_w",
    "key_body_height_err",
    "bike_component_pos_b",
  )
  # Patch dims that changed since training.
  merged["bicycle_joint_state"] = ObservationTermCfg(
    func=mdp.bicycle_joint_state_legacy4, params=cmd
  )
  merged["bicycle_joint_state_ref"] = ObservationTermCfg(
    func=mdp.bicycle_joint_state_ref_legacy4, params=cmd
  )
  merged["body_lin_vel"] = ObservationTermCfg(
    func=mdp.robot_body_lin_vel_b, params=cmd
  )
  merged["body_lin_vel_ref"] = ObservationTermCfg(
    func=mdp.robot_body_lin_vel_ref_b, params=cmd
  )
  merged["body_ang_vel"] = ObservationTermCfg(
    func=mdp.robot_body_ang_vel_b, params=cmd
  )
  merged["body_ang_vel_ref"] = ObservationTermCfg(
    func=mdp.robot_body_ang_vel_ref_b, params=cmd
  )
  merged["body_pos"] = ObservationTermCfg(func=mdp.robot_body_pos_b, params=cmd)
  merged["body_ori"] = ObservationTermCfg(func=mdp.robot_body_ori_b, params=cmd)
  merged["key_body_height_w"] = ObservationTermCfg(
    func=mdp.key_body_height_w_legacy4, params=cmd
  )
  merged["key_body_height_ref_w"] = ObservationTermCfg(
    func=mdp.key_body_height_ref_w_legacy4, params=cmd
  )
  merged["key_body_height_err"] = ObservationTermCfg(
    func=mdp.key_body_height_err_legacy4, params=cmd
  )
  merged["bicycle_ref_contact_mask"] = ObservationTermCfg(
    func=mdp.bicycle_ref_contact_mask, params=cmd
  )
  merged["foot_pedal_lin_vel_b"] = ObservationTermCfg(
    func=mdp.foot_pedal_lin_vel_b, params=cmd
  )
  merged["foot_pedal_lin_vel_ref_b"] = ObservationTermCfg(
    func=mdp.foot_pedal_lin_vel_ref_b, params=cmd
  )
  critic = {k: merged[k] for k in critic_keys if k in merged}
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic,
    concatenate_terms=True,
    enable_corruption=False,
  )


def disable_actor_observation_noise(cfg) -> None:
  """Turn off actor observation corruption (no additive noise)."""
  cfg.observations["actor"].enable_corruption = False


def disable_general_domain_randomization(cfg) -> None:
  """Turn off BeyondMimic-style general DR for ablations / bike debugging.

  Removes:
  - RSI noise: root pose / root vel / joint pos offsets at motion reset
  - Startup DR events: base CoM, encoder bias, foot friction
  - Interval push_robot
  - Actor observation corruption
  Does **not** touch bicycle curriculum assists (kinematic unlock, roll hold, …).
  """
  motion_cmd = cfg.commands.get("motion")
  if motion_cmd is not None:
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

  for name in ("push_robot", "base_com", "encoder_bias", "foot_friction"):
    cfg.events.pop(name, None)

  if "actor" in cfg.observations:
    cfg.observations["actor"].enable_corruption = False


def configure_ratchet_v3_dr_curriculum(cfg) -> None:
  """Add 25--50k core DR and 50--75k actuator/disturbance DR to V3."""
  robot_cfg = cfg.scene.entities["robot"]
  if robot_cfg.articulation is None:
    raise ValueError("Ratchet V3 DR requires an articulated robot")
  robot_cfg.articulation = copy.deepcopy(robot_cfg.articulation)
  robot_cfg.articulation.actuators = tuple(
    DelayedActuatorCfg(
      base_cfg=copy.deepcopy(actuator_cfg),
      delay_target="position",
      delay_min_lag=0,
      delay_max_lag=2,
      delay_update_period=0,
    )
    for actuator_cfg in robot_cfg.articulation.actuators
  )

  # Model/RSI DR is sampled before MotionCommand.reset on every reset.
  cfg.events["bicycle_v3_dr_reset"] = EventTermCfg(
    func=mdp.randomize_bicycle_v3_curriculum,
    mode="reset",
    params={"command_name": "motion"},
  )
  cfg.events["bicycle_v3_dr_push"] = EventTermCfg(
    func=mdp.apply_bicycle_dr_push,
    mode="step",
    params={"body_name": "torso_link"},
  )
  cfg.curriculum["bicycle_dr"] = CurriculumTermCfg(
    func=mdp.update_bicycle_dr_curriculum
  )

  # The V3 factory deliberately removed BeyondMimic corruption. Re-wrap only
  # the documented actor terms; the asymmetric critic remains noise-free.
  max_noise = {
    "motion_anchor_pos_b": 0.02,
    "motion_anchor_ori_b": 0.02,
    "base_lin_vel": 0.08,
    "base_ang_vel": 0.08,
    "joint_pos": 0.01,
    "joint_vel": 0.15,
  }
  actor = cfg.observations["actor"]
  for name, bound in max_noise.items():
    term = actor.terms.get(name)
    if term is None:
      continue
    source_func = term.func
    source_params = dict(term.params)
    term.func = mdp.curriculum_observation_noise
    term.params = {
      "source_func": source_func,
      "source_params": source_params,
      "max_abs": bound,
    }
    term.noise = None
  actor.enable_corruption = True


def configure_hybrid_dm_amp_ride20s(cfg, *, training: bool = True) -> None:
  """Use weak DeepMimic + conditional AMP in the extended 20 s ride."""
  ride_start, ride_end = 608, 1608
  for name in (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
  ):
    term = cfg.rewards[name]
    source_func = term.func
    source_params = dict(term.params)
    term.func = mdp.hybrid_phase_scaled_reward
    term.params = {
      "source_func": source_func,
      "source_params": source_params,
      "command_name": "motion",
      "ride_start_frame": ride_start,
      "ride_end_frame": ride_end,
      "ride_scale": 0.2,
      "blend_frames": 10,
    }

  cfg.rewards["bicycle_amp_style"] = RewardTermCfg(
    func=mdp.bicycle_amp_style_reward,
    weight=1.0,
    params={
      "cfg": AmpCfg(
        command_name="motion",
        command_speed=2.0,
        ride_start_frame=ride_start,
        ride_end_frame=ride_end,
        hidden_dims=(256, 128),
        grad_penalty=10.0,
        learning_rate=1.0e-4,
        weight_decay=1.0e-3,
        batch_size=256,
        buffer_steps=8,
        updates_per_step=1 if training else 0,
      )
    },
  )
  for group_name in ("actor", "critic"):
    terms = dict(cfg.observations[group_name].terms)
    terms["bicycle_speed_command"] = ObservationTermCfg(
      func=mdp.bicycle_fixed_speed_command, params={"speed": 2.0}
    )
    cfg.observations[group_name] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=cfg.observations[group_name].enable_corruption,
    )


def add_bicycle_freewheel(cfg) -> None:
  """Enable one-way crank→rear clutch each physics step (dynamic freewheel)."""
  cfg.events["bicycle_freewheel"] = EventTermCfg(
    func=mdp.apply_bicycle_freewheel,
    mode="step",
    params={"entity_name": "bicycle"},
  )


def add_t084_foot_ground_penalty(cfg) -> None:
  """Temporary T084: strong penalty for foot-floor touch during 6.0–9.5 s."""
  cfg.rewards["foot_ground_penalty_t084"] = RewardTermCfg(
    func=mdp.foot_ground_penalty_in_motion_window,
    weight=-30.0,
    params={
      "command_name": "motion",
      "sensor_name": "feet_ground_contact",
      "start_time_s": 6.0,
      "end_time_s": 9.5,
      "force_threshold": 10.0,
    },
  )


def bicycle_obs_dim() -> int:
  return 3 * len(BICYCLE_OBS_BODIES)
