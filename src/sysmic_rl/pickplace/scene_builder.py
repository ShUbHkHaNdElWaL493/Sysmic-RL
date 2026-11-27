"""Generate a scene MJCF that <include>s the (unmodified) robot file and adds a
table plus a pool of parked objects. The generated file is written next to the
robot XML so relative mesh paths keep working."""
from __future__ import annotations

import os
from pathlib import Path

from .config import SceneConfig
from .objects import make_object_specs, object_geoms_xml, _f


def build_scene_xml(cfg: SceneConfig) -> Path:
    robot = Path(cfg.robot_xml).resolve()
    if not robot.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {robot} (set SceneConfig.robot_xml or $SYSMIC_ROBOT_XML)")
    hx, hy = cfg.table_half_xy
    t = cfg.table_thickness
    cx, cy, top = (sum(cfg.table_x) / 2, sum(cfg.table_y) / 2, sum(cfg.table_z) / 2)
    # The table pose is re-written at every reset (model.body_pos); the pedestal is visual only and is resized then too.
    ped = (f'      <geom name="table_pedestal" type="box" size="0.03 0.03 {(top - t) / 2:.4f}" '
           f'pos="0 0 {-t - (top - t) / 2:.4f}" contype="0" conaffinity="0" rgba="0.4 0.3 0.2 1"/>')
    objs = []
    for i, name in enumerate(cfg.objects):
        park = (3.0 + 0.6 * i, 3.0, 0.002)
        objs.append(f'''    <body name="obj_{name}" pos="{_f(park)}">
      <freejoint name="obj_{name}_joint"/>
      {object_geoms_xml(name, cfg.object_scale, cfg.friction)}
    </body>''')
    xml = f'''<mujoco model="sysmic_pickplace">
  <include file="{robot.name}"/>
  <worldbody>
    <body name="table" pos="{cx} {cy} {top}">
      <geom name="table_top" type="box" size="{hx} {hy} {t / 2}" pos="0 0 {-t / 2}" rgba="0.6 0.45 0.3 1" friction="{_f(cfg.friction)}"/>
{ped}
    </body>
{chr(10).join(objs)}
  </worldbody>
</mujoco>
'''
    out = robot.parent / "scene_pickplace.generated.xml"
    if not out.exists() or out.read_text() != xml:
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, out)
    return out
