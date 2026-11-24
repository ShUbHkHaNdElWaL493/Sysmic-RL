# Sysmic pick-and-place: RL + imitation architecture

Goal: pick up and place plates, glasses, bowls (and mugs) with the Sysmic bimanual lift robot, trained with
RL that is bootstrapped from imitation learning. Your existing workspace (uv, MuJoCo, `mjcf/`, `src/sysmic_rl/`)
is untouched: everything is **additive** (new sub-packages, no edits to your robot MJCF or `SysmicV2Env`).

## 1. What the current env does and why it can't just be "retargeted"

| Finding in `SysmicV2Env` | Consequence for manipulation | What I did |
|---|---|---|
| Task = reach a *joint-sampled* EE pose; no objects, no table | Nothing to grasp | New scene (table + 4 object types) + new env |
| "EE" = `arm_*_wrist_roll_link` origin (variable named `jaw`) | Not the grasp point; offset ~5 cm from the fingers | `grasp_point()` = wrist-roll frame + calibrated offset (default = jaw COM) |
| Gripper is joint 6 with the same 0.02 rad/step scale as the arm | ~2 s to close; too slow | Per-joint scales (gripper 0.08) and lag caps |
| Joint `actuatorfrcrange=±3` N·m (actuator `forcerange=±11` is overridden) | Grip force is capped at ~3 N·m | Kept, but **check** that a 0.3 kg bowl can be held (probe + expert test) |
| Success bonus 1000, per-step penalties ~1e-3 | Poor scale for SAC critics | Dense staged reward, success = +50 |
| Observation not normalised | VecNormalize would desync BC and RL | Normalisation is inside the env (fixed ranges) |
| Mesh convex hulls as collision geoms on arms | Hull of a jaw ≠ a pad; grasps may be unrealistic | Flagged below; recommended primitive pads |
| Arm-vs-arm/links mesh overlap contacts at rest | Constant collision penalty | Contacts present at home pose are auto-ignored |

## 2. Target architecture

```
mjcf/sysmic_v2.xml (unchanged) ──<include>──► mjcf/scene_pickplace.generated.xml  (table + object pool, auto-generated)
                                                         │
src/sysmic_rl/pickplace/   config.py  objects.py  scene_builder.py  pickplace_env.py  probe.py
src/sysmic_rl/imitation/   expert.py (IK + phase machine)  collect.py (demos / DAgger)  dataset.py (HDF5)  bc.py
src/sysmic_rl/training/    common.py  train_sac.py  evaluate.py
tests/                     stub_robot.py (mesh-free robot)  test_pickplace.py
```

### 2.1 Scene
* Generated MJCF `<include>`s your robot file → your file is never edited. Written next to it so mesh paths resolve (git-ignore it).
* Object pool (plate, bowl, glass, mug) all present in the model, *parked on the floor far away*; per episode one is teleported to the
  table. This avoids recompiling the model per episode (fast, vectorisable).
* Hollow objects = ring of tilted boxes (compound), glass/mug body = cylinder. `condim=4` (torsional friction), elliptic cone,
  `impratio=10` (set in Python after compile) for stable grasps.
* Your `left_target` mocap sphere is reused as the goal marker.

### 2.2 Environment (`SysmicPickPlaceEnv`)
* **Action (13, unchanged layout)**: `[lift, left(5 joints + gripper), right(5 + gripper)]`, normalised joint-*delta* position targets,
  same EMA smoothing, anti-windup lag and sub-step interpolation as your env. The inactive arm is masked to 0.
  → a policy pre-trained on your reaching env has a compatible head.
* **Observation (86-D for 4 objects, layout in `env.obs_layout`)**: normalised joint pos/vel, grasp pose of both arms, previous action,
  object pose/velocity/type one-hot, object relative to each gripper, goal and goal-relative-to-object, active-arm one-hot,
  finger/jaw contact flags. State-based on purpose (see §6 for vision).
* **Episode**: random object, random arm (side with the object), random pose/yaw, goal ≥10 cm away on the same side,
  random friction ±30 % and mass ±20 %, random lift height and home noise.
* **Reward** (all weights in `RewardConfig`): `reach` (tanh of distance) + `grasp` (two-sided finger/jaw contact) + `lift` (gated by grasp)
  + `goal` (tanh distance object→goal, unlocked after first grasp) + `place` + `release` + `success(+50)`
  − collisions (capped) − action-rate − joint velocity − failure (−10 on fall/tip-over).
* **Termination**: success (object at goal ≤2 cm, upright, at rest, gripper released, held 10 steps), failure (fell off table, tilt > 1.1 rad),
  truncation at 400 steps (flagged `TimeLimit.truncated` for correct SB3 bootstrapping).

### 2.3 Imitation learning pipeline
1. **Scripted privileged expert** (`expert.py`): damped-least-squares Jacobian IK on the grasp point + tool-axis alignment
   (lift joint included, down-weighted), phases `pre → approach → close → lift → transport → lower → release → retreat`,
   with recovery (re-grasp) if the object is lost. It outputs actions in the policy's own action space, so labels are exact.
   Grasp style per object: `top` (pinch rim, plate/bowl) or `side` (glass/mug).
2. **Collect** (`collect.py`): execute expert + Gaussian noise (DART-style state diversity), store the *clean* expert action as the
   label, the executed action for RL replay. HDF5, one group per episode, only successful episodes by default.
3. **BC** (`bc.py`): the SB3 SAC actor *is* the BC network (loss on `tanh(mu)`; log-std pushed to −2 so exploration noise doesn't
   erase the clone). Episode-level holdout is easy to add; currently 10 % random-transition validation.
4. **DAgger** (`collect.py --policy … --beta`): policy drives, expert re-labels every visited state (cheap: expert is a function of sim state).
   Aggregate with `bc.py --demos a.h5 b.h5 …`.
5. **RL fine-tune** (`train_sac.py`): load BC actor → pre-fill replay with demos → critic-only warm-up (actor frozen) → SAC with
   low initial entropy coefficient (`auto_0.01`).

Why this combination: pure SAC on a 13-DoF, contact-rich, multi-object task with a 5-stage reward is sample-hungry; the expert gives
(a) a policy that already grasps, (b) successful transitions in the buffer, (c) a critic that has seen reward-bearing states before the actor is allowed to move.

## 3. Calibration — do this first (the defaults are educated guesses)

```bash
uv run python -m sysmic_rl.pickplace.probe --robot-xml mjcf/<your_robot>.xml
```
It prints: contacts at home pose, which gripper angle is open/closed, and the reachable grasp-point envelope with suggested
`table_top_z` / `spawn_x`. Then edit `SceneConfig`/`TaskConfig`:
* `table_top_z`, `spawn_x/y`, `table_center_xy`, `table_half_xy` — the arm's reach (~0.35 m from the shoulder) is short; the table
  must start in front of the base footprint and objects must sit inside the envelope.
* `gripper_open_q/close_q`, `arm_home` — from the gripper sweep.
* `ArmSpec.grasp_offset` / `tool_axis` — view with `env.render_mode="human"`; move until the grasp point is between the fingers.
* Object sizes (`object_scale`) vs jaw opening.
* Strongly recommended in MJCF: replace the jaw/finger mesh collision hulls by 2 box pads (friction ≥1) — convex hulls of fingers
  give unrealistic contacts and make the grasp-contact reward noisy.

## 4. Verification plan (in order — don't proceed until each passes)

| # | Check | Command | Pass criterion |
|---|---|---|---|
| 0 | Unit/smoke tests (mesh-free stub robot) | `uv run pytest` | all green |
| 1 | Geometry/gripper calibration | `probe` | table & objects inside envelope; open/closed known |
| 2 | Visual sanity | `evaluate --expert --render --episodes 3` | gripper straddles object, goal marker on table |
| 3 | Expert competence | `evaluate --expert --episodes 100` | ≥85 % overall, no object with <60 % (else tune `ExpertConfig`, grasp offsets) |
| 4 | Reward sanity | log `info["reward_terms"]` on expert episodes | return of successes ≫ failures; no single term dominates |
| 5 | Demo data | `collect --episodes 300 --noise 0.1` | ≥200 successes, balanced per object |
| 6 | BC | `bc` then `evaluate --model runs/bc/bc_sac.zip` | val MSE plateaus; rollout success ≥40 % (expect a gap to the expert → DAgger) |
| 7 | DAgger | 2–3 rounds × 100 eps, β = 0.5→0.2 | BC success approaches expert |
| 8 | RL | `train_sac`, TensorBoard | eval success ≥ BC at step 0 and not collapsing early (if it collapses: lower lr, longer `--warmup`, lower `ent_init`) |
| 9 | Generalisation | eval with unseen seeds, `object_scale` 0.9/1.1, friction range widened | success drop <15 pts |
| 10 | Ablation | RL from scratch vs BC-init vs BC-init+demos | BC-init should reach target success in a fraction of steps |

## 5. Run-book

```bash
uv add stable-baselines3 h5py tensorboard
uv add --dev pytest
echo -e "mjcf/*.generated.xml\ndata/\nruns/" >> .gitignore

uv run pytest
uv run python -m sysmic_rl.pickplace.probe
uv run python -m sysmic_rl.training.evaluate --expert --episodes 20 --render
uv run python -m sysmic_rl.imitation.collect --episodes 300 --noise 0.1 --out data/demos.h5
uv run python -m sysmic_rl.imitation.bc --demos data/demos.h5 --out runs/bc
uv run python -m sysmic_rl.training.evaluate --model runs/bc/bc_sac.zip
uv run python -m sysmic_rl.imitation.collect --episodes 100 --policy runs/bc/bc_sac.zip --beta 0.3 --out data/dagger1.h5
uv run python -m sysmic_rl.imitation.bc --demos data/demos.h5 data/dagger1.h5 --out runs/bc2
uv run python -m sysmic_rl.training.train_sac --demos data/demos.h5 data/dagger1.h5 --init runs/bc2/bc_sac.zip --out runs/sac
tensorboard --logdir runs
```

## 6. Known limitations / next steps
* **Verified here:** env, scene generation, expert plumbing, HDF5, probe, collect — on a mesh-free stub robot (7 tests pass).
  **Not verified here:** expert success on your real meshes (needs the calibration in §3), and the SB3/torch modules
  (`bc.py`, `train_sac.py`, `common.py` compile but were not run — torch didn't fit in my sandbox). Expect small API fixes.
* Single `n_envs` (demo pre-fill assumes it). For throughput, move to a custom RLPD/SACfD loop (50/50 demo/online sampling) with `SubprocVecEnv`.
* Base is fixed (wheels unused). Mobile manipulation = add base-velocity actions + a larger table region.
* Bimanual tasks (handover, carrying a tray) = unmask both arms, add a second goal; the 13-D action space already supports it.
* State-based obs assume perfect object pose. For deployment add a wrist/head camera, render RGB (`render_mode="rgb_array"`),
  and distil the state policy into an image policy (DAgger with the privileged expert is exactly the tool).
* Glass/mug are solid cylinders (no liquid, no handle-aware grasp); mug handle is a box.
