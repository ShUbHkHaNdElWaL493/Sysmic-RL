# `sysmic_rl.pickplace` — the world

This package contains everything about the *simulated world*: what is in the scene, what the robot can do,
what it can sense, and how well it is doing (reward). It has **no learning code** and does not import torch,
so you can run and test it on its own.

```
pickplace/
├── __init__.py         registers the Gym id "SysmicPickPlace-v0" and re-exports the main classes
├── config.py           all tunable numbers, as dataclasses
├── objects.py          procedural geometry for plate, bowl, glass, mug
├── scene_builder.py    writes a scene XML that includes your robot + table + objects
├── pickplace_env.py    SysmicPickPlaceEnv — the Gymnasium environment
└── probe.py            calibration report for your robot's geometry
```

Quick use:

```python
from sysmic_rl.pickplace import SysmicPickPlaceEnv, PickPlaceConfig

cfg = PickPlaceConfig()
cfg.scene.robot_xml = "mjcf/sysmic_v2.xml"       # or set $SYSMIC_ROBOT_XML
env = SysmicPickPlaceEnv(cfg, render_mode="human")
obs, info = env.reset(seed=0, options={"object": "glass", "arm": "left"})
while True:
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

## `config.py` — every knob in one place

**Why it exists:** so you change numbers in one file instead of hunting through the env. Four dataclasses are
nested into `PickPlaceConfig(scene, task, reward)`. Override fields in code (`cfg.task.max_episode_steps = 300`).

### `ArmSpec` / `LEFT_ARM` / `RIGHT_ARM` / `ARMS`
Describes how each arm is named in your MJCF and where its *grasp point* is.

| Field | Meaning |
|---|---|
| `joints` | 6 joint names: pan, shoulder, elbow, wrist_flex, wrist_roll, **gripper** (in this order — code relies on it) |
| `finger_body` | the body whose frame the grasp point is attached to (`arm_*_wrist_roll_link`, which also carries the fixed finger) |
| `jaw_body` | the moving jaw body (used to detect "jaw touches object") |
| `grasp_offset` | grasp point **in the finger body's frame**, metres. Default = the jaw's centre of mass. **A guess — calibrate it.** |
| `tool_axis` | direction the tool "points", in the finger body frame (default +x). The expert aligns this with down (−z) or forward (+x) |

### `SceneConfig`
| Field | Default | Meaning |
|---|---|---|
| `robot_xml` | `mjcf/sysmic_v2.xml` (or `$SYSMIC_ROBOT_XML`) | your robot file |
| `objects` | plate, glass, bowl, mug | pool used in training. **Order and length change the observation size** |
| `object_scale` | 1.0 | uniformly scales all objects |
| `friction` | (1, 0.005, 0.0001) | sliding, torsional, rolling friction on objects/table |
| `table_top_z`, `table_center_xy`, `table_half_xy`, `table_thickness` | 0.72, (0.32, 0), (0.20, 0.45), 0.04 | table pose/size (world frame = robot base frame) |
| `spawn_x`, `spawn_y` | (0.22, 0.34), (−0.25, 0.25) | where objects and goals are sampled |

### `TaskConfig`
| Group | Fields | Notes |
|---|---|---|
| Episode | `arm_mode` (`"left"`, `"right"`, `"auto"`), `max_episode_steps` (400), `frame_skip` (10) | `frame_skip × timestep(0.002 s)` = control period (20 ms) |
| Smoothing | `ema_alpha` (0.25) | lower = smoother but laggier |
| Speeds | `arm_action_scale` (0.03 rad/step), `gripper_action_scale` (0.08), `lift_action_scale` (0.002 m/step) | max target movement per step at action = 1 |
| Anti-windup | `arm_max_lag`, `gripper_max_lag`, `lift_max_lag` | target may not get further than this from the actual joint |
| Gripper | `gripper_open_q` (1.2), `gripper_close_q` (−0.1) | **guess — probe tells you which sign is open** |
| Home | `arm_home`, `home_noise`, `lift_home_range` | start pose of every episode |
| Success | `place_tol_xy` (2 cm), `place_tol_z` (1.5 cm), `place_max_tilt`, `place_max_speed`, `hold_steps` (10) | all must hold simultaneously for `hold_steps` |
| Failure | `fail_tilt` (1.1 rad) | object tipped over |
| Shaping | `lift_target` (0.08 m) | lift height that earns full lift reward |
| Randomisation | `randomize_physics`, `friction_scale`, `mass_scale`, `min_goal_dist` | |

### `RewardConfig`
Weights of each reward term; see [Reward](#reward) below.

---

## `objects.py` — tableware geometry

**Why:** real dish meshes aren't needed; simple primitives are fast and stable in contact.

* `ObjectSpec`: name, `height`, `radius`, `mass`, `grasp_style` (`"top"` = pinch the rim from above, `"side"` = wrap the body), `rim_radius`, colour.
* `make_object_specs(scale)`: returns the dict of specs. **The expert and the env read their dimensions from here**, so keep these numbers in sync with the geometry below.
* `object_geoms_xml(name, scale, friction)`: returns the `<geom>` XML.
  * plate/bowl = flat base cylinder + a ring of 16 tilted boxes (the rim/wall)
  * glass = solid cylinder
  * mug = solid cylinder + a box handle on the local +y side
* Object frame origin = **centre of the base bottom**, +z up. That's why placing on the table is simply `z = table_top_z`.

### Adding a new object (e.g. a "cup")
1. In `make_object_specs` add `"cup": ObjectSpec("cup", height, radius, mass, "side", 0.0, rgba)`.
2. In `object_geoms_xml` add an `elif name == "cup":` branch that appends `_geom(...)` strings (bottom at z = 0).
3. Add `"cup"` to `SceneConfig.objects`.
4. Run `uv run pytest` — `test_objects_settle_on_table` is parametrised over names; add `"cup"` to its list.
5. **Re-collect demos and retrain** (the observation grew by one one-hot entry).

---

## `scene_builder.py` — generate the scene file

`build_scene_xml(SceneConfig) -> Path` writes `scene_pickplace.generated.xml` **next to your robot XML** and returns its path.

* **Why a generated file that `<include>`s your robot?** So your robot MJCF is never edited, and mesh paths (relative to its folder) keep working.
* Contents: the include, a `table` body (static slab + visual legs), and one free-body per object parked on the floor far from the robot (x ≈ 3 m).
* **Why park instead of spawn?** Changing the object per episode by rebuilding the model is slow. Parked objects cost nothing; `reset()` just teleports the chosen one onto the table.
* Written atomically only when content changed (safe with several processes). Git-ignored.

---

## `pickplace_env.py` — the environment

`SysmicPickPlaceEnv(cfg=None, render_mode=None)` follows the Gymnasium API (`reset`, `step`, `render`, `close`).

### Spaces
* **Action**: `Box(−1, 1, (13,))` = `[lift, left×6, right×6]`. In each arm the 6th entry is the gripper.
* **Observation**: `Box(−inf, inf, (86,))` (with 4 objects). Layout (also `env.obs_layout[name] → slice`):

| Name | Size | Contents |
|---|---|---|
| `q_norm` | 13 | joint positions scaled to [−1, 1] by their limits (lift, left, right) |
| `qvel` | 13 | joint velocities ÷ 5 |
| `grasp_pos` | 6 | grasp point xyz of left, right |
| `grasp_quat` | 8 | finger-body orientation of left, right |
| `prev_action` | 13 | the smoothed previous action |
| `obj_pos`, `obj_quat` | 3, 4 | active object pose |
| `obj_linvel`, `obj_angvel` | 3, 3 | object velocities |
| `obj_type` | 4 | one-hot of which object it is |
| `obj_rel_ee` | 6 | object centre minus each grasp point |
| `goal_pos`, `goal_rel_obj` | 3, 3 | goal position and goal minus object |
| `active_arm` | 2 | one-hot left/right |
| `contacts` | 2 | fixed finger touches object, jaw touches object (0/1) |

### `reset(seed, options)` — what it does
1. Reset MuJoCo, pick object (random or `options["object"]`) and arm (`options["arm"]`, else `TaskConfig.arm_mode`; `"auto"` = random).
2. Set home pose (+ small noise), random lift height.
3. Sample object xy inside `spawn` (on the active arm's side of y=0) and a goal ≥ `min_goal_dist` away.
4. Teleport the object onto the table, randomise its friction (±30 %) and mass (±20 %).
5. Simulate 30 steps so it settles, move the red goal marker, return the first observation.

Other options: `object_pos=(x,y)`, `goal_pos=(x,y)`, `object_yaw`. Handy for debugging a specific case:
```python
env.reset(seed=1, options={"object": "plate", "arm": "right", "object_pos": (0.3, -0.1), "goal_pos": (0.25, -0.2)})
```

### `step(action)` — see the pipeline in [ARCHITECTURE.md §4](ARCHITECTURE.md#4-what-happens-during-one-environment-step)
Returns `(obs, reward, terminated, truncated, info)`.

### Reward
Computed every step in `_reward_and_flags()`; the individual terms are returned in `info["reward_terms"]`.

| Term | Formula (simplified) | Purpose |
|---|---|---|
| `reach` | `1·(1 − tanh(8·dist(gripper, object)))` | guide the hand to the object |
| `grasp` | `1` if finger **and** jaw touch object | reward actually holding it |
| `lift` | `2·clip(height_gain/0.08, 0, 1)` while grasped | reward raising it |
| `goal` | `3·(1 − tanh(6·dist(object, goal)))` after the first grasp | guide it to the goal |
| `place` | `2` if at goal, upright, nearly still | reward correct final pose |
| `release` | `1` if placed and gripper not touching | encourage letting go |
| `success` | `50` once placed & released for 10 steps | big bonus; ends episode |
| `collision` | `−min(3, 0.5·n_contacts + 20·penetration)` | avoid hitting table/other arm |
| `action_rate` | `−0.002·Σ(Δaction²)` | smooth motion |
| `velocity` | `−0.0005·Σ(qvel²)` | avoid wild swings |
| `fail` | `−10` on failure | object fell off / tipped |

**Why `tanh` shaping?** `1 − tanh(k·d)` goes smoothly from 0 (far) to 1 (at target): a gradient everywhere, bounded value. Bigger `k` = reward only
kicks in when closer.

**Collision handling:** contacts between arm geoms and anything except the active object are "bad", with exceptions: finger/jaw touching the
table (allowed for plates), and any pair that was already touching at the home pose (mesh overlap artefacts; computed once in
`_compute_baseline_contacts()`).

### Termination
* `terminated = success or failed`. Success = object within `place_tol_xy`/`place_tol_z` of goal, tilt < `place_max_tilt`, speed < `place_max_speed`, gripper released, for `hold_steps` consecutive steps. Failure = object fell below the table or tilt > `fail_tilt`.
* `truncated` = hit `max_episode_steps` without terminating. `info["TimeLimit.truncated"]` mirrors it so stable-baselines3 bootstraps correctly.

### `info` keys
`success`, `is_success` (same; SB3 convention), `failed`, `grasped`, `lifted`, `placed`, `released`, `err_reach`, `err_goal_xy`, `tilt`,
`n_bad_contacts`, `object`, `arm`, `reward_terms`.

### Helper methods used by other modules
`grasp_point(side)`, `tool_axis_world(side)`, `object_pos()`, `object_com()`, `object_tilt()`, `is_grasping()`, `joint_pos()`, `arm_dofs(side)`, and attributes
`target`, `act_scale`, `action_mask`, `sl` (slices for `left`/`right`), `lift_dof`, `finger_bid`, `goal_pos`, `spec`, `side`.
The **expert** relies on these — if you rename one, update `imitation/expert.py`.

### Internals worth knowing
* `13`-vectors (`target`, `lo`, `hi`, `act_scale`, …) are ordered `[lift, left×6, right×6]`.
* `elliptic` friction cone and `impratio=10` are set after model compile for stable grasping.
* Inactive arm: `action_mask` is 0, so its target never moves.

---

## `probe.py` — calibration report <a name="calibration"></a>

```bash
uv run python -m sysmic_rl.pickplace.probe --robot-xml mjcf/<robot>.xml --samples 20000
```

**Why:** the defaults for table height, spawn area, gripper angles and grasp offsets are guesses made without
seeing your meshes. The probe measures your robot instead of guessing. It prints three sections:

1. **Home-pose contacts** — body pairs overlapping at rest (auto-ignored by the reward).
2. **Gripper sweep** — distance jaw↔finger at the two joint limits. The *smaller* distance is "closed". Put those angles into `gripper_open_q` / `gripper_close_q`.
3. **Reachable envelope** — random collision-free configurations at lift 0 and lift max: x/y/z percentiles of the grasp point, and a suggested `table_top_z` and `spawn_x`.
   Treat suggestions as a starting point (the x-band can include points behind the robot; use only the part in front of the base).

Then *look* at it: `env = SysmicPickPlaceEnv(cfg, render_mode="human")`, reset, and check the grasp point (use a short script printing `env.grasp_point()` while you drag joints in the viewer) sits between the fingers; adjust `ArmSpec.grasp_offset` until it does.

---

## Making changes safely

| Change | How | Watch out for |
|---|---|---|
| Different table/spawn | edit `SceneConfig` | objects must be inside the reach envelope |
| New reward term | add to `terms` dict in `_reward_and_flags()` + weight in `RewardConfig` | demos store old rewards → re-collect |
| New observation | add `("name", array)` to `_obs_parts()` | dims change: obs_layout updates automatically, but all demos/models are invalid |
| Harder task | e.g. widen `spawn_*`, add `object_yaw` randomisation, raise `friction_scale` range | expect lower success; use curriculum (start easy) |
| New randomisation | in `reset()` after the physics block | keep it seeded via `self.np_random` (never `np.random`) so runs are reproducible |
| Both arms at once | set `action_mask` to all ones; add second object/goal | expert and reward only handle one arm today |

Rules of thumb:
* **Always use `self.np_random`** for randomness inside the env.
* After any change run `uv run pytest`, then the expert (`evaluate --expert`) — it is your regression test.
* Keep the reward **bounded** (a few units per step); huge terms destabilise SAC's critic.
