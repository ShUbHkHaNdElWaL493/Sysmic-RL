# `sysmic_rl.pickplace` — the world

This package contains everything about the *simulated world*: what is in the scene, where objects start and where they must end,
what the robot can sense, and how well it is doing (reward). It has **no learning code** and does not import torch,
so you can run and test it on its own.

```
pickplace/
├── __init__.py         registers the Gym id "SysmicPickPlace-v0" and re-exports the main classes
├── config.py           all tunable numbers, as dataclasses (+ OBJECT_SPECS: utensil dimensions)
├── pickplace_env.py    SysmicPickPlaceEnv — the Gymnasium environment (the heart)
└── probe.py            calibration report for your robot's geometry
```

The scene itself is the static file `mjcf/sysmic_v2_scene.xml` (see [Scene XML](#scene-xml)).

Quick use:

```python
from sysmic_rl.pickplace import SysmicPickPlaceEnv, PickPlaceConfig

cfg = PickPlaceConfig()
cfg.scene.scene_xml = "mjcf/sysmic_v2_scene.xml"   # the default; or set $SYSMIC_SCENE_XML
env = SysmicPickPlaceEnv(cfg, render_mode="human")
obs, info = env.reset(seed=0, options={"object": "plate"})      # plate -> info["arm"] == "both"
while True:
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The task in one picture

```
 ALL utensils START in a row   ──►  the active one is carried by     ──►  PLACED on a table
 on the robot's shelf               ONE arm or BOTH arms                   (new x, y, height, yaw every episode,
 (shuffled order/spacing/yaw)       (plate, bowl: both; glass, mug: one)    filtered for reachability)
```

---

## <a name="scene-xml"></a>Scene XML — `mjcf/sysmic_v2_scene.xml`

One static file, compiled once per env. It `<include>`s `sysmic_v2.xml` (never edited) and adds:

| Element | What | Rewritten at every `reset()` |
|---|---|---|
| site `shelf` | usable top of the shelf on `base_link` (z = 0.496, 13.5 × 45.5 cm) | — (read once) |
| geom `shelf_top` | invisible (group 3) collision box whose top face is the shelf surface | — |
| `<contact><exclude>` base_link ↔ each utensil | MuJoCo collides the `base_link` mesh as its **convex hull**, which fills the shelf recess; the utensils ignore the hull and rest on `shelf_top` instead | — |
| body `table` (geom `table_top` + 4 visual legs) | destination table; body origin = centre of the top face | `model.body_pos` (x, y, height) + `model.body_quat` (yaw) |
| bodies `obj_plate/glass/bowl/mug` (free joints) | utensils, origin = centre of the base bottom | `data.qpos` (row on the shelf) |

The legs are long enough for the highest table and simply sink below the (opaque) floor for lower tables, so no geom is resized at runtime.
Utensil dimensions used by the code live in `config.OBJECT_SPECS` — keep them in sync with the XML geometry.

**Adding a utensil:** add an `obj_<name>` body with a free joint (and an `<exclude>` against `base_link`) to the XML, its spec to `OBJECT_SPECS`,
the name to `SceneConfig.objects`, then **re-collect demos and retrain** (the observation grows). All utensils must fit in a row on the shelf.

---

## `config.py` — every knob in one place

Dataclasses nested into `PickPlaceConfig(scene, task, reward)`. Override fields in code (`cfg.task.max_episode_steps = 300`).

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
| `scene_xml` | `mjcf/sysmic_v2_scene.xml` (or `$SYSMIC_SCENE_XML`) | the scene file |
| `objects` | plate, glass, bowl, mug | utensils (must exist in the XML). **Order and length change the observation size** |
| `shelf_site`, `shelf_min_gap` | `"shelf"`, 0.01 | site the row is laid out in; min free space between neighbours |
| `table_x`, `table_y`, `table_z` | (0.35,0.65), (−0.25,0.25), (0.55,1.05) | ranges the table **centre** and **top height** are sampled from each episode |
| `table_yaw` | (−π/2, π/2) | table yaw range (covers every orientation of a rectangle). Table size comes from the XML |
| `table_clearance` | 0.03 | the table must be at least this far from every robot geom (`base_link` = its convex hull) |
| `reach_check` | True | filter object/table poses with the reach cloud (disable to skip the ~1 s start-up cost) |
| `reach_samples` | 50 000 | random arm configurations per arm in the reach cloud |
| `reach_tol` | 0.04 | grasp point must be within this distance of a reachable sample |
| `reach_axis_tol` | 0.5 rad | …and that sample's tool axis must be this close to the required one |

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

## `pickplace_env.py` — the environment
`SysmicPickPlaceEnv(cfg=None, render_mode=None)` follows the Gymnasium API.

### Spaces
* **Action** `Box(−1, 1, (13,))` = `[lift, left×6, right×6]` (6th entry of each arm = gripper). Inactive arms are masked to 0.
* **Observation** `Box(−inf, inf, (103,))` with 4 utensils. Layout in `env.obs_layout[name] → slice`:

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
| `others_pos` | 9 | positions of the 3 inactive utensils (obstacles), in `objects` order |

### `grasp_plan()` — the central geometric helper
`env.grasp_plan(obj_pos=None, sides=None) -> {side: (grasp_point, tool_axis)}` is used by the reward, observation, reach filter and expert.
* One arm, `top` object: rim point on the robot-facing side, tool **down**. One arm, `side` object: body at `side_height_frac`, tool **forward** (+x).
* Both arms: left hand on the **+y** side, right hand on the **−y** side, at radius `rim_radius` (top) or `radius` (side).

### <a name="bimanual"></a>Bimanual episodes
* Chosen per episode if the object is in `bimanual_objects` and `rng < bimanual_prob` (or force with `reset(options={"arm": "both"})`; `"left"`/`"right"` force one arm).
* The active utensil is put in a middle slot of the shelf row and must satisfy `|y| ≤ bimanual_y_max`, so both hands can reach opposite sides.
* "Grasped" = **every** active arm has finger **and** jaw contact. Reward `reach` = mean distance of both hands to their planned points.
* `bimanual_style="down"`: both tools point down onto opposite rims (pinch from above). `"inward"`: tools point at each other horizontally (squeeze from the sides). Try the other if one never reaches.
* `info["arm"] == "both"`; `evaluate.py` reports it as `arm:both`.

### `reset(seed, options)`
1. Choose object and arm(s); set home pose (random lift, small joint noise).
2. **Shelf row:** all utensils in shuffled order along the shelf, random spacing (Dirichlet split of the spare width), x jitter where the
   shelf depth allows (none for plate/bowl), random yaw (the mug's handle stays pointing front/back). Up to 100 tries until the active
   utensil's grasp plan is reachable (and |y| small enough if bimanual).
3. **Table:** sample centre/height/yaw in `table_*`; reject if it is within `table_clearance` of the robot (checked with `mj_geomDistance`:
   the table and `base_link` are both world-welded, so MuJoCo would never report a contact between them) or the goal's grasp plan is
   unreachable; the goal is sampled in the table frame, ≥ `min_goal_dist` from the object.
4. Randomise friction/mass of the active utensil; write all utensil poses; settle 30 steps; move the red goal marker.
5. If nothing feasible is found: `info["reach_fallback"]=True` (and a `RuntimeWarning` for the table) — your ranges don't match the arms' reach.

Options: `object`, `arm`, `goal_pos=(x,y)`, `table_pos=(x,y,z)` (+ `table_yaw`). Example:
```python
env.reset(seed=1, options={"object": "plate", "arm": "both", "table_pos": (0.45, 0.0, 0.8), "table_yaw": 0.3, "goal_pos": (0.40, 0.05)})
```

### `step(action)` — see [README §4](../../../README.md#4-what-happens-during-one-environment-step)

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
| `collision` | `−min(3, 0.5·n + 20·penetration)` | avoid hitting the table, the other arm and the **other utensils** |
| `action_rate`, `velocity` | small quadratic penalties | smooth motion |
| `fail` | `−10` on failure | object fell / tipped |

Allowed contacts (not penalised): any arm geom with the **active** utensil; fingertips with the **table or the shelf/base**; any pair already touching at the home pose.
Touching an inactive utensil is a collision. Failure: object below `min(table top, shelf height) − 0.25 m`, or tilt > `fail_tilt`.

### Helper methods used by other modules
`grasp_point(side)`, `tool_axis_world(side)`, `grasp_plan()`, `reachable(plan, sides)`, `seed_config(goals)`, `object_pos()`, `is_grasping()`, `joint_pos()`,
`arm_dofs(side)`, `table_to_world(uv)` / `world_to_table(xy)`, attributes `sides`, `target`, `act_scale`, `action_mask`, `sl`, `lift_dof`, `finger_bid`, `goal_pos`, `shelf`, `spec`.
The **expert** relies on them — update `imitation/expert.py` if you rename one.

### Reachability (why episodes are always feasible)
At start-up the env samples `reach_samples` random (lift, 5 joints) configurations per arm and stores grasp point, tool axis and joints (kinematics only, ~1 s).
`reachable()` asks: is there a sample whose grasp point is within `reach_tol` of the plan point **and** whose tool axis is within `reach_axis_tol` of the required one?
`seed_config()` returns the best sample as a starting pose for the expert. This is a *necessary-ish* test (it ignores collisions and joint dynamics), not a proof.

---

## `probe.py` — calibration report <a name="calibration"></a>

```bash
uv run python -m sysmic_rl.pickplace.probe --samples 20000      # [--scene-xml mjcf/sysmic_v2_scene.xml]
```
Sections:
1. **Shelf** — the site rectangle and how far below the arm bases it is at lift 0 / max.
2. **Reachability** — per utensil: % of shelf spawns reachable with one arm and with both arms, % of random table poses clear of the robot, and % that also have a reachable goal.
   Low numbers tell you what to change: widen/shift `table_*`, give the object a different `grasp_style`, fix `grasp_offset`/gripper angles, relax `reach_axis_tol`.
3. **Home-pose contacts** (auto-ignored by the reward). 4. **Gripper sweep** — smaller jaw↔finger distance = closed → set `gripper_open_q/close_q`. 5. **Reachable envelope** for hand-tuning.

After the probe, *look* at the scene (`render_mode="human"`) and move `ArmSpec.grasp_offset` until the grasp point sits between the fingers.

---

## Making changes safely

| Change | How | Watch out for |
|---|---|---|
| Different table range | `SceneConfig.table_*` | re-run probe; check "goal reachable" |
| Different table size / shelf | edit `table_top` / the `shelf` site + `shelf_top` geom in the scene XML | verify with probe + viewer |
| New reward term | add to `terms` in `_reward_and_flags()` + `RewardConfig` | demos store old rewards → re-collect |
| New observation | `_obs_parts()` | dims change → demos/models invalid |
| More bimanual objects | `TaskConfig.bimanual_objects` | re-collect; check probe "both-arms" % |
| Three-point / tray grasp | extend `grasp_plan()` (and the expert uses it automatically) | action space already supports it |
| New randomisation | in `reset()` using `self.np_random` | never use global `np.random` |

Rules of thumb: always use `self.np_random`; after any change run `uv run pytest` then `evaluate --expert`; keep rewards bounded (a few units/step).
