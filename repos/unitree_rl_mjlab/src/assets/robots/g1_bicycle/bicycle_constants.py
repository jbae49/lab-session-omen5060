"""GMR articulated bicycle asset for mjlab scenes."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg
from src import SRC_PATH

BICYCLE_XML: Path = SRC_PATH / "assets" / "gmr" / "bicycle" / "g1_bicycle.xml"
assert BICYCLE_XML.exists()

BICYCLE_SADDLE60_XML: Path = (
  SRC_PATH / "assets" / "gmr" / "bicycle" / "g1_bicycle_saddle60.xml"
)
assert BICYCLE_SADDLE60_XML.exists()

BICYCLE_SADDLE58_XML: Path = (
  SRC_PATH / "assets" / "gmr" / "bicycle" / "g1_bicycle_saddle58.xml"
)
assert BICYCLE_SADDLE58_XML.exists()


def get_spec(
  *,
  dynamic: bool = False,
  freewheel: bool | None = None,
  xml_path: Path | None = None,
  collision_condim: int | None = None,
  handlebar_friction: float | None = None,
) -> mujoco.MjSpec:
  """Load bicycle MjSpec.

  ``dynamic``: reserved flag (actuators are always stripped for RL).
  ``freewheel``: if True (default when ``dynamic``), remove the fixed-gear
  equality so a one-way clutch event can couple crank→rear only when driving.
  ``xml_path``: optional override (e.g. saddle60); default is ``BICYCLE_XML``.
  """
  if freewheel is None:
    freewheel = dynamic
  path = xml_path if xml_path is not None else BICYCLE_XML
  spec = mujoco.MjSpec.from_file(str(path))
  if collision_condim is not None:
    if collision_condim not in (1, 3, 4, 6):
      raise ValueError(f"Unsupported MuJoCo condim: {collision_condim}")
    for geom in spec.geoms:
      if geom.group == 3 and geom.contype != 0:
        geom.condim = int(collision_condim)
  if handlebar_friction is not None:
    if handlebar_friction < 0.0:
      raise ValueError("handlebar_friction must be non-negative")
    for name in (
      "handlebar_col_left",
      "handlebar_col_right",
      "handlebar_thick_col",
    ):
      geom = spec.geom(name)
      if geom is not None:
        geom.friction = [float(handlebar_friction), 0.005, 0.001]
  # Terrain entity provides the ground plane in mjlab tracking tasks.
  try:
    spec.delete(spec.geom("floor"))
  except Exception:
    pass
  # Actuators removed; legs / contacts (or freewheel clutch) drive the bike.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  if freewheel:
    for eq in list(spec.equalities):
      spec.delete(eq)
  return spec


def get_g1_bicycle_cfg(
  *,
  dynamic: bool = False,
  freewheel: bool | None = None,
  xml_path: Path | None = None,
  collision_condim: int | None = None,
  handlebar_friction: float | None = None,
) -> EntityCfg:
  if freewheel is None:
    freewheel = dynamic
  return EntityCfg(
    spec_fn=lambda: get_spec(
      dynamic=dynamic,
      freewheel=freewheel,
      xml_path=xml_path,
      collision_condim=collision_condim,
      handlebar_friction=handlebar_friction,
    ),
    articulation=None,
  )
