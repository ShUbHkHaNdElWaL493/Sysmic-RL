# `sysmic_rl.env` — the world

This package contains everything about the *simulated world*: what is in the scene, where objects start and where they must end,
what the robot can sense, and how well it is doing (reward). It has **no learning code** and does not import torch,
so you can run and test it on its own.

```
env/
├── __init__.py         registers the Gym id "SysmicBot-v1" and re-exports the main classes
├── config.py           all tunable numbers, as dataclasses (+ OBJECT_SPECS: utensil dimensions)
└── env.py              SysmicBotEnv — the Gymnasium environment (the heart)
```

The scene itself is the static file `mjcf/sysmic_v2_scene.xml` (see [Scene XML](#scene-xml)).

Quick use:

```python
from sysmic_rl.env import SysmicBotEnv, SysmicBotConfig

cfg = SysmicBotConfig()
cfg.scene.scene_xml = "mjcf/sysmic_v2_scene.xml"   # the default; or set $SYSMIC_SCENE_XML
env = SysmicBotEnv(cfg, render_mode="human")
obs, info = env.reset(seed=0, options={"object": "plate"})      # plate -> info["arm"] == "both"
while True:
   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The task in one picture

```
 ALL utensils START in a row   ──►  the active one is carried by     ──►  PLACED on a table
 on the shelf riser                 ONE arm or BOTH arms                   in front (new x, y, yaw every episode,
 (shuffled order/spacing/yaw)       (plate, bowl: both; glass, mug: one)    filtered for reachability)
```

---

## <a name="scene-xml"></a>Scene XML — `mjcf/sysmic_v2_scene.xml`

One static file, compiled once per env. It `<include>`s `sysmic_v2.xml` (never edited) and adds:

| Element | What | Rewritten at every `reset()` |
|---|---|---|
| site `shelf` | usable top of the riser (z = 0.646, 13.5 × 45.5 cm) | — (read once) |
| geom `shelf_top` | 15 cm riser standing on the robot's shelf (z = 0.496); the utensils rest on it (the robot's shelf alone is below the arms' reach) | — |
| `<contact><exclude>` base_link ↔ each utensil | MuJoCo collides the `base_link` mesh as its **convex hull**, which fills the shelf recess; the utensils ignore the hull and rest on `shelf_top` instead | — |
| body `table` (geom `table_top` + 4 visual legs) | destination table; body origin = centre of the top face; top at a fixed 0.85 m (= `table_z`) | `model.body_pos` (x, y) + `model.body_quat` (yaw) |
| body `goal` (mocap, visual sphere) | red goal marker, 3 cm above the goal | `data.mocap_pos` |
| bodies `obj_plate/glass/bowl/mug` (free joints) | utensils, origin = centre of the base bottom (details below) | `data.qpos` (row on the shelf) |
| `<option><flag multiccd>` | several contact points between convex shapes, so the utensil bases rest flat instead of rocking/sinking | — |
| more `<contact><exclude>` | robot links two joints apart whose convex hulls overlap although the real parts never touch (wrist↔jaw, arm base↔upper arm, shoulder pan↔forearm) | — |

**Utensils.** Each has a smooth visual mesh (`mjcf/meshes/utensils/*.obj`, group 2, no collision) and invisible primitive
collision geoms (group 3) that keep the real shape — a mesh would collide as its convex hull, filling the hollow parts and the handle's hole.
Masses come from the collision geoms.

| Utensil | Shape | Grasped at |
|---|---|---|
| plate | parabolic dish, Ø 12.4 cm, rim 2.2 cm high, 3 mm wall, 0.20 kg | rim, both arms |
| bowl | elliptical bowl with a foot, Ø 13 cm × 5 cm, 3 mm wall, 0.25 kg | rim, both arms |
| glass | wine glass: Ø 6 cm foot, 9 mm × 5.8 cm stem, hollow 7 cm bowl, 14 cm tall, 0.15 kg | **stem**, one arm |
| mug | hollow, 8 mm solid base, Ø 7.2 × 8.5 cm, handle loop (1 cm bar, 1.9 × 3.5 cm hole) on local +x, 0.28 kg | **handle** (outer bar), one arm |

Utensil dimensions used by the code live in `config.OBJECT_SPECS` (incl. the grasp point: `grasp_r`, `grasp_z`, `wall_slope`) — keep them in sync with the XML geometry.

**Adding a utensil:** add an `obj_<name>` body with a free joint (and an `<exclude>` against `base_link`) to the XML, its spec to `OBJECT_SPECS`,
the name to `SceneConfig.objects`, then **re-collect demos and retrain** (the observation grows). All utensils must fit in a row on the shelf.

---

## `config.py` — every knob in one place

Dataclasses nested into `SysmicBotConfig(scene, task, reward)`. Override fields in code (`cfg.task.max_episode_steps = 300`).

### `ArmSpec` / `LEFT_ARM` / `RIGHT_ARM` / `ARMS`
| Field | Meaning |
|---|---|
| `joints` | 6 joint names: pan, shoulder, elbow, wrist_flex, wrist_roll, **gripper** (in this order — code relies on it) |
| `finger_body` | body whose frame the grasp point is attached to (`arm_*_wrist_roll_link`, carries the fixed finger) |
| `jaw_body` | the moving jaw body (used to detect "jaw touches object") |
| `grasp_offset` | grasp point **in the finger body's frame**: where a ~1 cm object rests against the fixed finger when clamped, ~1.7 cm behind the tip (0.120, ±0.0027, 0.0047) |
| `tool_axis` | direction the fingers point, in the finger body frame = the wrist-roll axis (+x) |
| `close_axis` | direction the jaw moves when closing, in the finger body frame: (0, ∓0.5, −0.866). The fixed finger is on the opposite side of the object |

All three were **calibrated once** from the robot's collision shapes (the robot MJCF is never edited, so they don't change).
`calibrate_gripper()` in `tests/test_env.py` re-derives them — a 1 cm rod slid against the fixed finger's hull, then the jaw closed
until it touches the rod (q = 1.52 < `gripper_close_q`, so it squeezes) — and `test_gripper_calibration_matches_config` fails if they
drift. The grasp point is ~6 mm from the wrist-roll axis, so the wrist roll turns the closing direction without moving the grasp point.

### `SceneConfig`
| Field | Default | Meaning |
|---|---|---|
| `scene_xml` | `mjcf/sysmic_v2_scene.xml` (or `$SYSMIC_SCENE_XML`) | the scene file |
| `objects` | plate, glass, bowl, mug | utensils (must exist in the XML). **Order and length change the observation size** |
| `shelf_site`, `shelf_min_gap` | `"shelf"`, 0.01 | site the row is laid out in; min free space between neighbours |
| `table_z` | 0.85 | constant table top height (must match the XML) |
| `table_gap`, `table_y` | (0, 0.08), (−0.15, 0.15) | nearest table edge this far beyond the front limit (shelf front + `table_clearance`); table centre y |
| `goal_margin` | 0.01 | min distance of the placed object's footprint from the table edge |
| `table_yaw` | (−π/2, π/2) | table yaw range (covers every orientation of a rectangle). Table size comes from the XML |
| `table_clearance` | 0.03 | the table must be at least this far from every robot geom (`base_link` = its convex hull) |
| `reach_check` | True | filter object/table poses with the reach cloud (disable to skip the ~10 s start-up cost) |
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
| Gripper | `gripper_open_q` (−0.1), `gripper_close_q` (1.75) | q increases = closing (jaw↔finger 0.059 m at −0.17, 0.035 m at 1.75); checked by `test_gripper_close_q_is_closed` |
| Home | `arm_home`, `home_noise`, `lift_home_range` | start pose of every episode |
| Grasp pitch | `rim_pitch` (0.9), `stem_pitch` (1.0), `handle_pitch` (0.7) rad | tool axis pitched down from horizontal per grasp style (the heading follows the arm) |
| **Bimanual** | `bimanual_objects` (plate, bowl), `bimanual_y_max` (0.10), `bimanual_rim_angle` (0.8 rad) | see [Bimanual episodes](#bimanual) |
| Success | `place_tol_xy` (2 cm), `place_tol_z`, `place_max_tilt`, `place_max_speed`, `hold_steps` (10) | all must hold together for `hold_steps` |
| Failure / shaping | `fail_tilt`, `lift_target` (0.08 m) | |
| Randomisation | `randomize_physics`, `friction_scale`, `mass_scale` | |

### `RewardConfig`
Weights of each reward term; see [Reward](#reward).

---

## `env.py` — the environment
`SysmicBotEnv(cfg=None, render_mode=None)` follows the Gymnasium API.

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
`env.grasp_plan(obj_pos=None, sides=None, obj_yaw=None) -> {side: (grasp_point, tool_axis, close_axis)}` is used by the reward,
observation, reach filter and expert. The arms have 5 joints (shoulder, elbow and wrist-flex bend in one plane), so the tool **heading is
not free**: for rim/stem grasps only the pitch is prescribed and the heading is the arm's (`heading()`; `grasp_axes(side, tool_now)` keeps the
current one). The reach check compares only the pitch for those.
* `rim` (plate, bowl): point on the rim wall (`grasp_r`, `grasp_z`); pitch `rim_pitch`; the jaw closes across the wall **from the outside**
  (the thick fixed finger goes inside the dish). One arm: robot-facing point; both arms: `bimanual_rim_angle` either side of it (left +y, right −y).
* `stem` (glass): point on the stem axis at `grasp_z`; pitch `stem_pitch`; the jaws close horizontally across the stem.
* `handle` (mug): the handle's outer bar; the tool points along the handle towards the body (pitch `handle_pitch`) and the jaws close on the
  loop's flat sides. This fixes the full tool direction, so `reset()` aims the handle at the grasping arm (`yaw_to_arm()`), and the reach
  check at the goal assumes it stays aimed at the arm while carried.

### <a name="bimanual"></a>Bimanual episodes
* **Always** for objects in `bimanual_objects`, never for the others. `reset(options={"arm": ...})` only picks the side of a single-arm
  object; asking for the wrong mode raises `ValueError`.
* The active utensil is put in a middle slot of the shelf row and must satisfy `|y| ≤ bimanual_y_max`, so both hands can reach opposite sides.
* "Grasped" = **every** active arm has finger **and** jaw contact. Reward `reach` = mean distance of both hands to their planned points.
* Smaller `bimanual_rim_angle` = hands further from the neighbouring utensils on the shelf, but closer to each other (the reach check ignores hand↔hand collisions).
* `info["arm"] == "both"`; `evaluate.py` reports it as `arm:both`.

### `reset(seed, options)`
1. Choose object and arm(s); set home pose (random lift, small joint noise).
2. **Shelf row:** all utensils in shuffled order along the shelf, random spacing (Dirichlet split of the spare width), x jitter where the
   shelf depth allows (none for plate/bowl), random yaw (the mug's handle is aimed at the grasping arm, ±0.15 rad). Up to 100 tries until the active
   utensil's grasp plan is reachable (and |y| small enough if bimanual).
3. **Table:** sample yaw and centre y, put the nearest edge `table_gap` beyond `front_x`, top at `table_z`; reject if it is within `table_clearance` of the robot (checked with `mj_geomDistance`:
   the table and `base_link` are both world-welded, so MuJoCo would never report a contact between them) or the goal's grasp plan is
   unreachable (up to 60 table poses × 20 goals); the goal is sampled in the table frame, `goal_margin` inside the edge.
4. Randomise friction/mass of the active utensil; write all utensil poses; settle 30 steps; move the red goal marker.
5. If nothing feasible is found: `info["reach_fallback"]=True` (and a `RuntimeWarning` for the table) — your ranges don't match the arms' reach.

Options: `object`, `arm` (side of a single-arm object), `goal_pos=(x,y)`, `table_pos=(x,y)` or `(x,y,z)` (+ `table_yaw`). Example:
```python
env.reset(seed=1, options={"object": "plate", "table_pos": (0.55, 0.0), "table_yaw": 0.3, "goal_pos": (0.36, 0.0)})
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
`grasp_point(side)`, `tool_axis_world(side)`, `close_axis_world(side)`, `grasp_plan()`, `grasp_axes(side, tool_now)`, `close_signed()`,
`reachable(plan, sides)`, `seed_config(goals)`, `object_pos()`, `object_yaw()`, `is_grasping()`, `joint_pos()`,
`arm_dofs(side)`, `table_to_world(uv)` / `world_to_table(xy)`, attributes `sides`, `target`, `act_scale`, `action_mask`, `sl`, `lift_dof`, `finger_bid`, `goal_pos`, `shelf`, `spec`.
The **expert** relies on them — update `imitation/expert.py` if you rename one.

### Reachability (why episodes are always feasible)
At start-up the env samples `reach_samples` random (lift, 5 joints) configurations per arm and keeps those where the arm touches neither
the robot, itself, the other arm (at home) nor the riser (~65 % survive; ~10 s). Each sample stores grasp point, tool axis, closing axis and joints.
`reachable()` asks: is there a sample whose grasp point is within `reach_tol` of the plan point **and** whose tool axis (pitch only for rim/stem)
is within `reach_axis_tol`? The closing direction is free (wrist roll). `seed_config()` returns the best sample, with the wrist roll turned so
the jaws close the right way, as a starting pose for the expert. Still a *necessary-ish* test: utensils, the table and the path there are ignored.

---

## <a name="calibration"></a>Workspace calibration (done; how to re-check)

The workspace is set up for the default scene and needs no calibration step before training:

| What | Where it lives | Re-checked by |
|---|---|---|
| Gripper geometry (`grasp_offset`, `tool_axis`, `close_axis`) and open/close angles | `ArmSpec`, `TaskConfig.gripper_*_q` | `test_gripper_calibration_matches_config`, `test_gripper_close_q_is_closed` |
| Shelf / table / goals reachable for every utensil | scene XML + `SceneConfig.table_*` + grasp pitches | every `reset()` (`info["reach_fallback"]`, `RuntimeWarning`) and `test_default_workspace_is_reachable` |
| Utensils spawn without overlapping each other or the robot | `OBJECT_SPECS.footprint_y`, `shelf_min_gap` | `test_default_workspace_is_reachable` |
| Rest-pose mesh overlaps | computed automatically at env start (currently none) | — |

Measured on the default scene (200 random poses each): shelf spawns reachable — plate 95 %, glass 100 %, bowl 100 %, mug 44 %;
random table poses with a reachable goal 15–46 %. `reset()` retries (100 shelf rows, 60 tables × 20 goals), so 300 test resets had **0** fallbacks.

---

## Making changes safely

| Change | How | Watch out for |
|---|---|---|
| Different table range | `SceneConfig.table_*` | `uv run pytest -k workspace`; `reach_fallback` in `evaluate` output |
| Different table size / height / riser | edit `table_top` (+ `table_z`) / the `shelf` site + `shelf_top` geom in the scene XML | `uv run pytest -k workspace` + viewer |
| New reward term | add to `terms` in `_reward_and_flags()` + `RewardConfig` | demos store old rewards → re-collect |
| New observation | `_obs_parts()` | dims change → demos/models invalid |
| More bimanual objects | `TaskConfig.bimanual_objects` | re-collect; `uv run pytest -k workspace` |
| Three-point / tray grasp | extend `grasp_plan()` (and the expert uses it automatically) | action space already supports it |
| New randomisation | in `reset()` using `self.np_random` | never use global `np.random` |

Rules of thumb: always use `self.np_random`; after any change run `uv run pytest` then `evaluate --expert`; keep rewards bounded (a few units/step).
