# `sysmic_rl.pickplace` — the world

This package contains everything about the *simulated world*: what is in the scene, where objects start and where they must end,
what the robot can sense, and how well it is doing (reward). It has **no learning code** and does not import torch,
so you can run and test it on its own.

```
pickplace/
├── __init__.py         registers the Gym id "SysmicPickPlace-v0" and re-exports the main classes
├── config.py           all tunable numbers, as dataclasses
├── objects.py          procedural geometry for plate, bowl, glass, mug
├── scene_builder.py    writes a scene XML: includes your robot + a spawnable table + parked objects
├── surface.py          finds the robot's own flat "extrusion" (where objects start) by ray casting
├── pickplace_env.py    SysmicPickPlaceEnv — the Gymnasium environment (the heart)
└── probe.py            calibration report for your robot's geometry
```

Quick use:

```python
from sysmic_rl.pickplace import SysmicPickPlaceEnv, PickPlaceConfig

cfg = PickPlaceConfig()
cfg.scene.robot_xml = "mjcf/sysmic_v2.xml"       # or set $SYSMIC_ROBOT_XML
env = SysmicPickPlaceEnv(cfg, render_mode="human")
obs, info = env.reset(seed=0, options={"object": "plate"})      # plate -> info["arm"] == "both"
while True:
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The task in one picture

```
 objects START on the robot's own flat extrusion  ──►  carried by ONE arm or BOTH arms  ──►  PLACED on a table
 (auto-detected; may ride on the lift carriage)        (plate, bowl: both; glass, mug: one)    (new pose every episode, always reachable)
```

---

## `config.py` — every knob in one place

Dataclasses nested into `PickPlaceConfig(scene, task, reward, source)`. Override fields in code (`cfg.task.max_episode_steps = 300`).

### `ArmSpec` / `LEFT_ARM` / `RIGHT_ARM` / `ARMS`
| Field | Meaning |
|---|---|
| `joints` | 6 joint names: pan, shoulder, elbow, wrist_flex, wrist_roll, **gripper** (in this order — code relies on it) |
| `finger_body` | body whose frame the grasp point is attached to (`arm_*_wrist_roll_link`, carries the fixed finger) |
| `jaw_body` | the moving jaw body (used to detect "jaw touches object") |
| `grasp_offset` | grasp point **in the finger body's frame**, metres. Default = the jaw's centre of mass. **A guess — calibrate it.** |
| `tool_axis` | direction the tool "points", in the finger body frame (default +x). The expert aligns it with down / forward / inward |

### `SceneConfig`
| Field | Default | Meaning |
|---|---|---|
| `robot_xml` | `mjcf/sysmic_v2.xml` (or `$SYSMIC_ROBOT_XML`) | your robot file |
| `objects` | plate, glass, bowl, mug | object pool. **Order and length change the observation size** |
| `object_scale`, `friction` | 1.0, (1, 0.005, 0.0001) | object size multiplier; sliding/torsional/rolling friction |
| `table_half_xy`, `table_thickness` | (0.18, 0.22), 0.04 | destination-table size |
| `table_x`, `table_y`, `table_z` | (0.25,0.55), (−0.25,0.25), (0.55,1.05) | ranges the table **centre** and **top height** are sampled from each episode |
| `table_clearance` | 0.03 | the table must be at least this far from every robot geom |
| `reach_check` | True | filter object/table poses with the reach cloud (disable to skip the ~1 s start-up cost) |
| `reach_samples` | 50 000 | random arm configurations per arm in the reach cloud |
| `reach_tol` | 0.04 | grasp point must be within this distance of a reachable sample |
| `reach_axis_tol` | 0.5 rad | …and that sample's tool axis must be this close to the required one |

### `SourceConfig` — where objects start
| Field | Default | Meaning |
|---|---|---|
| `manual` | None | `(cx, cy, z, hx, hy)` world rectangle at the home pose. `None` = **auto-detect** by ray casting |
| `body` | None | force detection onto one body (e.g. `"lift_carriage_link"`), or attach a manual rectangle to it |
| `search_x/y/z` | (−0.1,0.6) / (−0.45,0.45) / (0.3,1.4) | box searched for the surface |
| `step` | 0.01 | ray grid spacing (m) |
| `min_area` | 0.01 m² | smallest patch accepted |
| `edge_margin` | 0.015 | objects keep this far from the patch edge |

### `TaskConfig`
| Group | Fields | Notes |
|---|---|---|
| Episode | `arm_mode` (`"left"`/`"right"`/`"auto"`) — for single-arm episodes, `max_episode_steps` (500), `frame_skip` (10) | `auto` = side the object is on |
| Smoothing | `ema_alpha` (0.25) | lower = smoother but laggier |
| Speeds | `arm_action_scale` (0.03 rad/step), `gripper_action_scale` (0.08), `lift_action_scale` (0.002 m/step) | max target movement per step at action = 1 |
| Anti-windup | `arm_max_lag`, `gripper_max_lag`, `lift_max_lag` | target may not get further than this from the actual joint |
| Gripper | `gripper_open_q` (1.2), `gripper_close_q` (−0.1) | **guess — probe tells you which is open** |
| Home | `arm_home`, `home_noise`, `lift_home_range` | start pose of every episode |
| Grasp geometry | `side_height_frac` (0.6), `top_depth` (0.006) | where on the object the grasp point sits |
| **Bimanual** | `bimanual_objects` (plate, bowl), `bimanual_prob` (1.0), `bimanual_y_max` (0.10), `bimanual_style` (`"down"`/`"inward"`) | see [Bimanual episodes](#bimanual) |
| Success | `place_tol_xy` (2 cm), `place_tol_z`, `place_max_tilt`, `place_max_speed`, `hold_steps` (10) | all must hold together for `hold_steps` |
| Failure / shaping | `fail_tilt`, `lift_target` (0.08 m), `min_goal_dist` (0.10 m) | |
| Randomisation | `randomize_physics`, `friction_scale`, `mass_scale` | |

### `RewardConfig`
Weights of each reward term; see [Reward](#reward).

---

## `objects.py` — tableware geometry
* `ObjectSpec`: `height`, `radius` (outer footprint), `mass`, `grasp_style` (`"top"` pinch rim from above / `"side"` wrap the body), `rim_radius`, colour.
* `make_object_specs(scale)`: dict of specs. The env, reachability check and expert read dimensions from here — keep in sync with the geometry.
* `object_geoms_xml(name, scale, friction)`: plate/bowl = base disk + ring of 16 tilted boxes; glass = cylinder; mug = cylinder + box handle on local +y.
* Object frame origin = **centre of the base bottom**, +z up, so resting on a surface means `z = surface height`.

**Adding an object:** add the spec in `make_object_specs`, a branch in `object_geoms_xml`, the name in `SceneConfig.objects`,
add it to the parametrised test, then **re-collect demos and retrain** (the observation grows).

---

## `scene_builder.py`
`build_scene_xml(SceneConfig) -> Path` writes `scene_pickplace.generated.xml` next to your robot XML (so relative mesh paths work) and returns the path.
It `<include>`s your robot (never edited), adds a `table` body (slab + visual pedestal) and one free-body per object **parked far away**.
The table is moved and its pedestal resized at every reset (`model.body_pos`, `geom_size`), so there is no recompilation per episode.

---

## `surface.py` — finding the robot's "extrusion"
**Why:** you want objects to start on a feature of the robot itself, whose shape I can't see. So the env finds it instead of guessing.

How `detect_support_surface` works:
1. Hide arm, object and table geoms; cast rays straight down (`mj_ray`, collision geoms only) over a grid covering `SourceConfig.search_*`.
2. Group hits by height (5 mm bins); the **largest connected patch** (≥ `min_area`) wins. Body = the body most of its cells belong to.
3. Shrink the patch by `edge_margin`. Store cells in world coordinates at scan time (+ the body's position then).
4. `SupportSurface.sample(rng, radius, data)` returns a random point where a disk of that radius fits entirely on the patch (translated by however far the body has moved since the scan).
5. `env.surface_moves` is True if the surface body changes height with the lift joint (e.g. carriage).

Detection failing/choosing wrong? Run the probe, then set `SourceConfig.body="..."` (restrict to a body) or `SourceConfig.manual=(cx,cy,z,hx,hy)`.
Limitation: only translation of the body is tracked (fine for a slide-joint carriage, not for a rotating link).

---

## `pickplace_env.py` — the environment
`SysmicPickPlaceEnv(cfg=None, render_mode=None)` follows the Gymnasium API.

### Spaces
* **Action** `Box(−1, 1, (13,))` = `[lift, left×6, right×6]` (6th entry of each arm = gripper). Inactive arms are masked to 0.
* **Observation** `Box(−inf, inf, (94,))` with 4 objects. Layout in `env.obs_layout[name] → slice`:

| Name | Size | Contents |
|---|---|---|
| `q_norm` | 13 | joint positions scaled to [−1, 1] by limits |
| `qvel` | 13 | joint velocities ÷ 5 |
| `grasp_pos` | 6 | grasp point xyz of left, right |
| `grasp_quat` | 8 | finger-body orientation of left, right |
| `prev_action` | 13 | smoothed previous action |
| `obj_pos`, `obj_quat` | 3, 4 | active object pose |
| `obj_linvel`, `obj_angvel` | 3, 3 | object velocities |
| `obj_type` | 4 | one-hot object |
| `obj_rel_ee` | 6 | object centre − each grasp point |
| `plan_rel` | 6 | planned grasp point − current grasp point, per active arm (zeros for an inactive arm) |
| `goal_pos`, `goal_rel_obj` | 3, 3 | goal and goal − object |
| `active_arm` | 2 | `[left active, right active]` — `[1,1]` means a **bimanual** episode |
| `contacts` | 4 | `[left finger, left jaw, right finger, right jaw]` touching the object |

### `grasp_plan()` — the central geometric helper
`env.grasp_plan(obj_pos=None, sides=None) -> {side: (grasp_point, tool_axis)}` is used by the reward, observation, reach filter and expert.
* One arm, `top` object: rim point on the robot-facing side, tool **down**. One arm, `side` object: body at `side_height_frac`, tool **forward** (+x).
* Both arms: left hand on the **+y** side, right hand on the **−y** side, at radius `rim_radius` (top) or `radius` (side).

### <a name="bimanual"></a>Bimanual episodes
* Chosen per episode if the object is in `bimanual_objects` and `rng < bimanual_prob` (or force with `reset(options={"arm": "both"})`; `"left"`/`"right"` force one arm).
* Object spawn is limited to `|y| ≤ bimanual_y_max` so both hands can reach opposite sides.
* "Grasped" = **every** active arm has finger **and** jaw contact. Reward `reach` = mean distance of both hands to their planned points.
* `bimanual_style="down"`: both tools point down onto opposite rims (pinch from above). `"inward"`: tools point at each other horizontally (squeeze from the sides). Try the other if one never reaches.
* `info["arm"] == "both"`; `evaluate.py` reports it as `arm:both`.

### `reset(seed, options)`
1. Choose object and arm(s); set home pose (random lift, small joint noise).
2. **Object pose:** sample on the source surface (up to 300 tries) until the grasp plan is reachable (and |y| small enough if bimanual).
3. **Table:** sample centre/height in `table_*`; reject if it touches the robot (`table_clearance`) or the goal's grasp plan is unreachable; goal on the table, ≥ `min_goal_dist` from the object.
4. Randomise friction/mass; place the object on the surface; settle 30 steps; move the red goal marker.
5. If nothing feasible is found: `info["reach_fallback"]=True` (and a `RuntimeWarning` for the table) — your ranges don't match the arms' reach.

Options: `object`, `arm`, `object_pos=(x,y)`, `goal_pos=(x,y)`, `table_pos=(x,y,z)`, `object_yaw`. Example:
```python
env.reset(seed=1, options={"object": "plate", "arm": "both", "table_pos": (0.35, 0.0, 0.8), "goal_pos": (0.35, 0.05)})
```

### `step(action)` — see [ARCHITECTURE.md §4](ARCHITECTURE.md#4-what-happens-during-one-environment-step)

### Reward <a name="reward"></a>
Computed in `_reward_and_flags()`; terms returned in `info["reward_terms"]`.

| Term | Formula (simplified) | Purpose |
|---|---|---|
| `reach` | `1·(1 − tanh(8·mean dist(hand, planned grasp point)))` | guide the hand(s) to the grasp points |
| `grasp` | `1` if all active arms touch with finger and jaw | reward holding it |
| `lift` | `2·clip(height_gain/0.08, 0, 1)` while grasped | reward raising it |
| `goal` | `3·(1 − tanh(6·dist(object, goal)))` after first grasp | guide it to the goal |
| `place` | `2` if at goal, upright, nearly still | correct final pose |
| `release` | `1` if placed and no active gripper touching | let go |
| `success` | `50` after `hold_steps` of placed+released | big bonus; ends episode |
| `collision` | `−min(3, 0.5·n + 20·penetration)` | avoid hitting table/other arm |
| `action_rate`, `velocity` | small quadratic penalties | smooth motion |
| `fail` | `−10` on failure | object fell / tipped |

Allowed contacts (not penalised): any arm geom with the active object; fingertips with the **table or the source surface**; any pair already touching at the home pose.
Failure: object below `min(table top, source height) − 0.25 m`, or tilt > `fail_tilt`.

### Helper methods used by other modules
`grasp_point(side)`, `tool_axis_world(side)`, `grasp_plan()`, `reachable(plan, sides, source)`, `seed_config(goals)`, `object_pos()`, `is_grasping()`, `joint_pos()`,
`arm_dofs(side)`, attributes `sides`, `target`, `act_scale`, `action_mask`, `sl`, `lift_dof`, `finger_bid`, `goal_pos`, `surface`, `surface_moves`, `spec`.
The **expert** relies on them — update `imitation/expert.py` if you rename one.

### Reachability (why episodes are always feasible)
At start-up the env samples `reach_samples` random (lift, 5 joints) configurations per arm and stores grasp point, tool axis and joints (kinematics only, ~1 s).
`reachable()` asks: is there a sample whose grasp point is within `reach_tol` of the plan point **and** whose tool axis is within `reach_axis_tol` of the required one?
For objects on a carriage-mounted surface, samples are compared in the carriage frame (the lift height cancels out).
`seed_config()` returns the best sample as a starting pose for the expert. This is a *necessary-ish* test (it ignores collisions and joint dynamics), not a proof.

---

## `probe.py` — calibration report <a name="calibration"></a>

```bash
uv run python -m sysmic_rl.pickplace.probe --robot-xml mjcf/<robot>.xml --samples 20000
```
Sections:
1. **Source surface** — body, top height, extent, usable area, `moves_with_lift`. Check it is the extrusion you meant.
2. **Reachability** — per object: % of spawn spots reachable with one arm and with both arms, and % of random table poses that are collision-free *and* have a reachable goal.
   Low numbers tell you what to change: widen/shift `table_*`, give the object a different `grasp_style`, fix `grasp_offset`/gripper angles, relax `reach_axis_tol`.
3. **Home-pose contacts** (auto-ignored by the reward). 4. **Gripper sweep** — smaller jaw↔finger distance = closed → set `gripper_open_q/close_q`. 5. **Reachable envelope** for hand-tuning.

After the probe, *look* at the scene (`render_mode="human"`) and move `ArmSpec.grasp_offset` until the grasp point sits between the fingers.

---

## Making changes safely

| Change | How | Watch out for |
|---|---|---|
| Different table range/size | `SceneConfig.table_*` | re-run probe; check "table pose+goal ok" |
| Objects start elsewhere | `SourceConfig.manual/body` | verify with probe + viewer |
| New reward term | add to `terms` in `_reward_and_flags()` + `RewardConfig` | demos store old rewards → re-collect |
| New observation | `_obs_parts()` | dims change → demos/models invalid |
| More bimanual objects | `TaskConfig.bimanual_objects` | re-collect; check probe "both-arms" % |
| Three-point / tray grasp | extend `grasp_plan()` (and the expert uses it automatically) | action space already supports it |
| New randomisation | in `reset()` using `self.np_random` | never use global `np.random` |

Rules of thumb: always use `self.np_random`; after any change run `uv run pytest` then `evaluate --expert`; keep rewards bounded (a few units/step).
