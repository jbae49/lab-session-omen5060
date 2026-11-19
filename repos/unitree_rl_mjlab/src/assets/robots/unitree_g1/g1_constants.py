"""Unitree G1 constants."""

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()

G1_DHAND_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_dhand.xml"
)
assert G1_DHAND_XML.exists()

G1_DHAND_PELVIS_CAPSULE_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_dhand_pelvis_capsule.xml"
)
assert G1_DHAND_PELVIS_CAPSULE_XML.exists()

G1_DHAND_PELVIS_SPHERE_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_dhand_pelvis_sphere.xml"
)
assert G1_DHAND_PELVIS_SPHERE_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_dhand_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_DHAND_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_rubber_box_dhand_feet_spec(
  dhand_xml: Path | None = None,
) -> mujoco.MjSpec:
  """Dhand body with RubberBox hands and default G1 capsule feet."""
  dhand_root = ET.parse(dhand_xml or G1_DHAND_XML).getroot()
  rubber_root = ET.parse(G1_XML).getroot()
  dhand_assets = dhand_root.find("asset")
  rubber_assets = rubber_root.find("asset")
  assert dhand_assets is not None and rubber_assets is not None

  existing_meshes = {mesh.get("name") for mesh in dhand_assets.findall("mesh")}
  for mesh in rubber_assets.findall("mesh"):
    name = mesh.get("name", "")
    if "rubber_hand" in name and name not in existing_meshes:
      dhand_assets.append(copy.deepcopy(mesh))

  collision_default = next(
    node
    for node in dhand_root.iter("default")
    if node.get("class") == "collision"
  )
  if not any(node.get("class") == "foot_capsule" for node in collision_default):
    foot_capsule = ET.SubElement(collision_default, "default", {"class": "foot_capsule"})
    ET.SubElement(foot_capsule, "geom", {"size": "0.01", "type": "capsule"})

  for side in ("left", "right"):
    dhand_wrist = next(
      body
      for body in dhand_root.iter("body")
      if body.get("name") == f"{side}_wrist_yaw_link"
    )
    rubber_wrist = next(
      body
      for body in rubber_root.iter("body")
      if body.get("name") == f"{side}_wrist_yaw_link"
    )
    for geom in list(dhand_wrist.findall("geom")):
      name = geom.get("name", "")
      mesh = geom.get("mesh", "")
      if name.startswith(f"{side}_hand_plate_") or mesh == f"{side}_bicycle_hand_grip":
        dhand_wrist.remove(geom)
    for geom in rubber_wrist.findall("geom"):
      if (
        geom.get("mesh", "").startswith(f"{side}_rubber_hand")
        or geom.get("name", "").startswith(f"{side}_hand_")
      ):
        dhand_wrist.append(copy.deepcopy(geom))

    dhand_foot = next(
      body
      for body in dhand_root.iter("body")
      if body.get("name") == f"{side}_ankle_roll_link"
    )
    rubber_foot = next(
      body
      for body in rubber_root.iter("body")
      if body.get("name") == f"{side}_ankle_roll_link"
    )
    for geom in list(dhand_foot.findall("geom")):
      name = geom.get("name", "")
      if name.startswith(f"{side}_foot"):
        dhand_foot.remove(geom)
    for geom in rubber_foot.findall("geom"):
      if (geom.get("name") or "").startswith(f"{side}_foot"):
        dhand_foot.append(copy.deepcopy(geom))

  return mujoco.MjSpec.from_string(
    ET.tostring(dhand_root, encoding="unicode"), assets=get_assets("assets")
  )


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

# Dhand grip plates: sliding µ=1.0, same as the bicycle contact DR nominal.
# Thin foot-sole layers share foot friction/condim for pedal contact.
DHAND_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot\d+(_sole)?_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot\d+(_sole)?_collision$": 1},
  friction={
    r"^(left|right)_foot\d+(_sole)?_collision$": (1.0,),
    r"^(left|right)_hand_plate_.*_collision$": (1.0,),
  },
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot\d+_collision$": 1},
  friction={r"^(left|right)_foot\d+_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_dhand_robot_cfg(*, hand_friction: float = 1.0) -> EntityCfg:
  """G1 with ㄷ-shaped bicycle grip hand collisions (4 boxes per hand)."""
  if hand_friction < 0.0:
    raise ValueError("hand_friction must be non-negative")
  collision = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
      r"^(left|right)_foot\d+(_sole)?_collision$": 3,
      ".*_collision": 1,
    },
    priority={r"^(left|right)_foot\d+(_sole)?_collision$": 1},
    friction={
      r"^(left|right)_foot\d+(_sole)?_collision$": (1.0,),
      r"^(left|right)_hand_plate_.*_collision$": (float(hand_friction),),
      r"^pelvis_collision": (1.0,),
    },
  )
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=get_dhand_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_rubber_box_dhand_feet_robot_cfg(
  *,
  hand_friction: float = 1.0,
  dhand_xml: Path | None = None,
) -> EntityCfg:
  """G1 retaining Dhand feet/body geometry but using fitted rubber-box hands."""
  if hand_friction < 0.0:
    raise ValueError("hand_friction must be non-negative")
  collision = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
      r"^(left|right)_foot\d+(_sole)?_collision$": 3,
      r"^(left|right)_hand_.*_collision$": 3,
      r"^pelvis_collision": 3,
      ".*_collision": 1,
    },
    priority={
      r"^(left|right)_foot\d+(_sole)?_collision$": 1,
      r"^(left|right)_hand_.*_collision$": 1,
      r"^pelvis_collision": 1,
    },
    friction={
      r"^(left|right)_foot\d+(_sole)?_collision$": (1.0,),
      r"^(left|right)_hand_.*_collision$": (float(hand_friction),),
      r"^pelvis_collision": (1.0,),
    },
  )
  xml = dhand_xml
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(collision,),
    spec_fn=lambda xml=xml: get_rubber_box_dhand_feet_spec(dhand_xml=xml),
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
