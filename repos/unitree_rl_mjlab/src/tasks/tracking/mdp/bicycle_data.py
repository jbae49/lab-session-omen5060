"""Bicycle interaction metadata for tracking tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BicycleContactPair:
  """One end-effector interaction channel (bike part or ground)."""

  key: str
  ee_body: str
  bike_body: str
  sensor_name: str
  dist_threshold: float = 0.12
  target_mode: Literal["bike", "ground"] = "bike"
  # Optional offset in ``bike_body`` local frame (contact / reward target point).
  target_local_pos: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class BicycleContactEvent:
  """Enable or disable contact channels at ``time_s`` (state persists until changed)."""

  time_s: float
  active_keys: tuple[str, ...]
  enable: bool = True


# test_motion1_t084 at 50 fps (user-specified attach / detach frames).
#   220: hands + saddle ON
#   258: right pedal ON
#   340: left pedal ON
#   515: both pedals OFF
#   711: saddle OFF
#   790: both hands OFF
_T084_FPS = 50.0
T084_BICYCLE_CONTACT_SCHEDULE: tuple[BicycleContactEvent, ...] = (
  BicycleContactEvent(
    time_s=220 / _T084_FPS,
    active_keys=("handle_left", "handle_right", "saddle"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=258 / _T084_FPS,
    active_keys=("pedal_right",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=340 / _T084_FPS,
    active_keys=("pedal_left",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=515 / _T084_FPS,
    active_keys=("pedal_left", "pedal_right"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=711 / _T084_FPS,
    active_keys=("saddle",),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=790 / _T084_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=False,
  ),
)

# test_motion1_t045 at 50 fps (812 frames).
#   125: both hands ON
#   190: saddle ON
#   205: right pedal ON
#   286: left pedal ON
#   367: both pedals OFF
#   389: both feet–ground ON (until end)
#   580: saddle OFF
#   600: both hands OFF
_T045_FPS = 50.0
T045_BICYCLE_CONTACT_SCHEDULE: tuple[BicycleContactEvent, ...] = (
  BicycleContactEvent(
    time_s=125 / _T045_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=190 / _T045_FPS,
    active_keys=("saddle",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=205 / _T045_FPS,
    active_keys=("pedal_right",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=286 / _T045_FPS,
    active_keys=("pedal_left",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=367 / _T045_FPS,
    active_keys=("pedal_left", "pedal_right"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=389 / _T045_FPS,
    active_keys=("foot_left_ground", "foot_right_ground"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=580 / _T045_FPS,
    active_keys=("saddle",),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=600 / _T045_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=False,
  ),
)

# test_motion1_t060 at 50 fps (912 frames).
#   183: saddle ON
#   206: both hands ON
#   225: right pedal ON
#   310: left pedal ON
#   515: both pedals OFF
#   523: both feet–ground ON (until end)
#   635: saddle OFF
#   720: both hands OFF
_T060_FPS = 50.0
T060_BICYCLE_CONTACT_SCHEDULE: tuple[BicycleContactEvent, ...] = (
  BicycleContactEvent(
    time_s=183 / _T060_FPS,
    active_keys=("saddle",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=206 / _T060_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=225 / _T060_FPS,
    active_keys=("pedal_right",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=310 / _T060_FPS,
    active_keys=("pedal_left",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=515 / _T060_FPS,
    active_keys=("pedal_left", "pedal_right"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=523 / _T060_FPS,
    active_keys=("foot_left_ground", "foot_right_ground"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=635 / _T060_FPS,
    active_keys=("saddle",),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=720 / _T060_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=False,
  ),
)

# bike_052 C7 contact windows at 30 Hz playback indices (half-open [on, off)).
# Source: GMR/scripts/smplx_to_robot_contact_swap.py CONTACT.
# Prefer ``BIKE052_HOLOSOMA_50HZ_CONTACT_SCHEDULE`` for Holosoma→mjlab 50 Hz clips.
_BIKE052_FPS = 30.0
BIKE052_BICYCLE_CONTACT_SCHEDULE: tuple[BicycleContactEvent, ...] = (
  BicycleContactEvent(
    time_s=0.0,
    active_keys=("foot_left_ground", "foot_right_ground"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=80 / _BIKE052_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=147 / _BIKE052_FPS,
    active_keys=("saddle",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=181 / _BIKE052_FPS,
    active_keys=("foot_left_ground", "foot_right_ground"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=287 / _BIKE052_FPS,
    active_keys=("pedal_right",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=363 / _BIKE052_FPS,
    active_keys=("pedal_left",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=442 / _BIKE052_FPS,
    active_keys=("pedal_left", "pedal_right"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=535 / _BIKE052_FPS,
    active_keys=("saddle",),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=605 / _BIKE052_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=False,
  ),
)

# Holosoma bike_052 full @ 50 Hz (Viser frame stamps on T=1109 clip).
# L/R pedals are independent events (do not share a single pedal gate).
_BIKE052_HOLO_FPS = 50.0
BIKE052_HOLOSOMA_50HZ_CONTACT_SCHEDULE: tuple[BicycleContactEvent, ...] = (
  BicycleContactEvent(
    time_s=138 / _BIKE052_HOLO_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=264 / _BIKE052_HOLO_FPS,
    active_keys=("saddle",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=464 / _BIKE052_HOLO_FPS,
    active_keys=("pedal_right",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=608 / _BIKE052_HOLO_FPS,
    active_keys=("pedal_left",),
    enable=True,
  ),
  BicycleContactEvent(
    time_s=760 / _BIKE052_HOLO_FPS,
    active_keys=("pedal_left", "pedal_right"),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=910 / _BIKE052_HOLO_FPS,
    active_keys=("saddle",),
    enable=False,
  ),
  BicycleContactEvent(
    time_s=1015 / _BIKE052_HOLO_FPS,
    active_keys=("handle_left", "handle_right"),
    enable=False,
  ),
)

# EE contact reward group weights (L/R split so schedules can differ).
# handle 40, saddle 20, pedal 40; ground unused (0).
BICYCLE_EEF_CONTACT_PAIR_WEIGHTS: dict[str, float] = {
  "handle_left": 20.0,
  "handle_right": 20.0,
  "saddle": 20.0,
  "pedal_left": 20.0,
  "pedal_right": 20.0,
  "foot_left_ground": 0.0,
  "foot_right_ground": 0.0,
}

# Independent scheduled contact terms (not the HDMI averaged EEF blob).
BICYCLE_SPLIT_CONTACT_KEYS: tuple[str, ...] = (
  "handle_left",
  "handle_right",
  "saddle",
  "pedal_left",
  "pedal_right",
)
BICYCLE_SPLIT_CONTACT_REWARD_NAMES: dict[str, str] = {
  "handle_left": "bicycle_contact_handle_left",
  "handle_right": "bicycle_contact_handle_right",
  "saddle": "bicycle_contact_saddle",
  "pedal_left": "bicycle_contact_pedal_left",
  "pedal_right": "bicycle_contact_pedal_right",
}

# Contact / retarget points (g1_bicycle*_xml, same as GMR contact_swap).
# Handle bar is 57 cm along fork Y from +0.285 (left=0 cm) to -0.285 (right=57 cm).
# Hand targets sit 5 cm in from each end.  A 2x-diameter collision segment at
# 42–48 cm is named handlebar_thick_col so it is not a rewarded grab surface.
HANDLE_GRIP_L_LOCAL = (-0.0337, 0.285, 0.20)
HANDLE_GRIP_R_LOCAL = (-0.0337, -0.285, 0.20)
HANDLE_HAND_L_LOCAL = (-0.0337, 0.235, 0.20)  # 5 cm from left end
HANDLE_HAND_R_LOCAL = (-0.0337, -0.235, 0.20)  # 52 cm from left end
# Pedal collision box center in pedal_* body frame (see g1_bicycle XML).
# Pedal box center in pedal_* body frame. Platform is 10 cm wide (Y half=0.05)
# with center 7.5 cm outboard of crank tip → world centers ±0.15 (30 cm stance).
PEDAL_L_CONTACT_LOCAL = (0.0, 0.075, 0.0)
PEDAL_R_CONTACT_LOCAL = (0.0, -0.075, 0.0)
# Midfoot sole-pad center in ankle_roll_link (``*_foot4_sole_collision``).
FOOT_SOLE_CONTACT_LOCAL = (0.039, 0.0, -0.0335)
# Preferential forefoot band on the foot sole (reward gating / visualization).
# Global heel→toe uses the full sole envelope in ankle +X (heel=0, toe=1),
# not each strip's own length. Outermost strips foot1/foot7 are excluded.
FOOT_SOLE_MID_STRIP_IDS: tuple[int, ...] = (2, 3, 4, 5, 6)
# Forefoot reward band in heel→toe normalized coordinates (heel=0, toe=1).
# The bike_052 riding motion loads the pedal primarily through the toe-side
# half of the sole rather than the geometric midfoot.
FOOT_SOLE_HEEL_TOE_MID_S0 = 0.5
FOOT_SOLE_HEEL_TOE_MID_S1 = 0.9
# Full thin-sole envelope in the ankle-roll link's local +X direction.  The
# reward-eligible forefoot band is the same one shown by
# ``preview_contact_reward_collision_viser.py``.
FOOT_SOLE_HEEL_X_LOCAL = -0.064
FOOT_SOLE_TOE_X_LOCAL = 0.142
# Saddle_3 center on frame (saddle57); saddle60 uses SADDLE60_LOCAL_POS.
SADDLE_CONTACT_LOCAL = (0.2327, 0.0, 0.570)

BICYCLE_CONTACT_PAIRS: tuple[BicycleContactPair, ...] = (
  BicycleContactPair(
    key="handle_left",
    ee_body="left_wrist_yaw_link",
    bike_body="fork",
    sensor_name="contact_handle_left",
    dist_threshold=0.18,
    target_local_pos=HANDLE_HAND_L_LOCAL,
  ),
  BicycleContactPair(
    key="handle_right",
    ee_body="right_wrist_yaw_link",
    bike_body="fork",
    sensor_name="contact_handle_right",
    dist_threshold=0.18,
    target_local_pos=HANDLE_HAND_R_LOCAL,
  ),
  BicycleContactPair(
    key="saddle",
    ee_body="pelvis",
    bike_body="frame",
    sensor_name="contact_saddle",
    dist_threshold=0.15,
    target_local_pos=SADDLE_CONTACT_LOCAL,
  ),
  BicycleContactPair(
    key="pedal_left",
    ee_body="left_ankle_roll_link",
    bike_body="pedal_left",
    sensor_name="contact_pedal_left",
    dist_threshold=0.12,
    target_local_pos=PEDAL_L_CONTACT_LOCAL,
  ),
  BicycleContactPair(
    key="pedal_right",
    ee_body="right_ankle_roll_link",
    bike_body="pedal_right",
    sensor_name="contact_pedal_right",
    dist_threshold=0.12,
    target_local_pos=PEDAL_R_CONTACT_LOCAL,
  ),
  BicycleContactPair(
    key="foot_left_ground",
    ee_body="left_ankle_roll_link",
    bike_body="",
    sensor_name="contact_foot_left_ground",
    dist_threshold=0.08,
    target_mode="ground",
  ),
  BicycleContactPair(
    key="foot_right_ground",
    ee_body="right_ankle_roll_link",
    bike_body="",
    sensor_name="contact_foot_right_ground",
    dist_threshold=0.08,
    target_mode="ground",
  ),
)

BICYCLE_CONTACT_PAIR_KEYS: tuple[str, ...] = tuple(p.key for p in BICYCLE_CONTACT_PAIRS)

# First ON time (seconds) per pedal key from T084 schedule — used for pre-contact guidance.
T084_PEDAL_ON_TIME_S: dict[str, float] = {
  "pedal_right": 258 / _T084_FPS,
  "pedal_left": 340 / _T084_FPS,
}


def build_contact_flags(
  num_frames: int,
  fps: float,
  schedule: tuple[BicycleContactEvent, ...],
) -> np.ndarray:
  """Build (T, num_pairs) contact flags from a time schedule with ON/OFF events."""
  flags = np.zeros((num_frames, len(BICYCLE_CONTACT_PAIR_KEYS)), dtype=np.float32)
  active: set[str] = set()
  events = sorted(
    ((int(round(ev.time_s * fps)), ev) for ev in schedule),
    key=lambda x: x[0],
  )
  event_idx = 0
  for frame in range(num_frames):
    while event_idx < len(events) and events[event_idx][0] <= frame:
      ev = events[event_idx][1]
      if ev.enable:
        active.update(ev.active_keys)
      else:
        active.difference_update(ev.active_keys)
      event_idx += 1
    for key_idx, key in enumerate(BICYCLE_CONTACT_PAIR_KEYS):
      flags[frame, key_idx] = 1.0 if key in active else 0.0
  return flags


# Bicycle bodies exposed in policy observations (pos in robot anchor frame).
BICYCLE_OBS_BODIES: tuple[str, ...] = (
  "pedal_left",
  "pedal_right",
  "fork",
  "frame",
)

# Full bike articulation bodies for per-component velocity obs (matches motion npz).
BICYCLE_VEL_OBS_BODIES: tuple[str, ...] = (
  "frame",
  "fork",
  "rear_wheel",
  "front_wheel",
  "crank",
  "pedal_left",
  "pedal_right",
)

# Root body used for object (bicycle) tracking rewards.
BICYCLE_TRACK_BODY = "frame"

# Interaction points in parent-body local frame (see g1_bicycle.xml).
# Handle ≈ handlebar capsule midpoint on ``fork``; saddle ≈ saddle_3 center on ``frame``.
HANDLE_PARENT_BODY = "fork"
HANDLE_LOCAL_POS = (-0.0337, 0.0, 0.20)
SADDLE_PARENT_BODY = "frame"
SADDLE_LOCAL_POS = (0.2327, 0.0, 0.570)
SADDLE60_LOCAL_POS = (0.2327, 0.0, 0.600)
SADDLE58_LOCAL_POS = (0.2327, 0.0, 0.580)
PEDAL_BODIES_FOR_OBS: tuple[str, ...] = ("pedal_left", "pedal_right")

# bike_052 @ 50 Hz motion (contact windows originally @ 30 Hz, ~664–665 frames).
# Left-foot ground-push reward: only [300, 340] @30 Hz (user-specified stamp window).
BIKE052_LEFT_FOOT_GROUND_PUSH_START_30 = 300
BIKE052_LEFT_FOOT_GROUND_PUSH_END_30 = 340
BIKE052_GROUND_START_FRAME_50 = int(
  round(BIKE052_LEFT_FOOT_GROUND_PUSH_START_30 * 50.0 / 30.0)
)  # 500
BIKE052_GROUND_END_FRAME_50 = int(
  round(BIKE052_LEFT_FOOT_GROUND_PUSH_END_30 * 50.0 / 30.0)
)  # 567
BIKE052_PELVIS_PUSH_START_S = 11.0
BIKE052_PELVIS_PUSH_END_S = 15.0
BIKE052_PELVIS_PUSH_START_FRAME_50 = int(round(11.0 * 50.0))  # 550
BIKE052_PELVIS_PUSH_END_FRAME_50 = int(round(15.0 * 50.0))  # 750
BIKE052_WARMUP_DECAY_ITERS = 30_000
