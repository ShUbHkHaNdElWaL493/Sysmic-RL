# `sysmic_rl.imitation` — the teacher

This package turns "a human-written program that can do the task" into "a neural network that can do the task".

```
imitation/
├── expert.py    scripted privileged expert: Jacobian IK + a phase state machine
├── dataset.py   save / load demonstrations (HDF5)
├── collect.py   run the expert (optionally mixed with a policy = DAgger) and record a dataset
└── bc.py        behaviour cloning: supervised training of the SAC actor on a dataset
```

## Why imitation learning at all?

RL from scratch must discover *by luck* that closing the gripper around a rim while lifting gives reward.
With 13 action dimensions that can take millions of steps — or never happen. Imitation gives the network a head
start: it already knows roughly what a good grasp looks like, so RL only has to refine it.

| Method | What it does | Strength | Weakness |
|---|---|---|---|
| **Behaviour cloning (BC)** | supervised: predict the expert's action from the observation | simple, fast, stable | errors compound; the policy drifts into states the expert never visited |
| **DAgger** | policy drives, expert labels the states the policy actually visits | fixes the drift | needs the expert to be queryable at any state (ours is, because it reads simulator state) |
| **Demo-seeded RL** | RL whose replay buffer starts with expert data | improves beyond the expert | needs RL machinery → see [training/README.md](../training/README.md) |

---

## `expert.py` — the scripted expert

### What it is
A hand-written controller that **cheats**: it reads the exact object pose, goal, contacts from the simulator
(the neural network only gets the 103-number observation). It handles **one arm or both** (plates/bowls are carried bimanually). It outputs actions in the **same 13-D normalised space**
as the policy, so its outputs are directly usable as supervised labels.

### How it works
1. **Targets:** `env.grasp_plan()` gives, per active arm, the grasp point and tool axis (the same plan the reward uses).
   Pre-grasp = grasp point pulled back along the tool axis by `pre_dist`.
   * targets = `env.grasp_plan()` (point, tool axis, closing axis): plate/bowl rim with both arms, glass stem, mug handle;
   * the IK servos the full gripper orientation: tool axis **and** closing direction (`env.grasp_axes()` keeps the arm's heading);
   * the joint-space seed (`env.seed_config`) already turns the wrist roll so the jaws close the right way.
2. **Seed** (`SEED` phase, global IK): purely local IK can get stuck at joint limits (e.g. tool pointing up). So the expert first asks
   `env.seed_config()` for the best sampled arm configuration for the pre-grasp pose and moves there **in joint space**.
3. **Move** (`_goto`): one step of *damped-least-squares IK over all active arms at once*.
   * Stacks, per arm, 3 position rows + 3 orientation rows (rotation about the tool axis ignored → wrist roll stays free). Unknowns: lift + 5 joints per arm.
   * Both arms share the lift joint, so solving them together avoids conflicting lift commands.
   * Converts `dq` into the normalised action `(q_now + dq − q_target)/scale`, clipped to [−1, 1].
3. **Phases** (state machine):

| Phase | Goal | Moves on when |
|---|---|---|
| `SEED` | joint-space move to a sampled configuration near the pre-grasp pose | all joints within `seed_tol` or timeout |
| `PRE` | go to pre-grasp pose with gripper open | position error < 1.5×`pos_tol` and axis error < `axis_tol`, or timeout |
| `APPROACH` | move into the grasp pose | error < `pos_tol` or timeout |
| `CLOSE` | close gripper(s) | ≥ `close_steps` and **every** active arm has finger+jaw contact (or 4× timeout) |
| `LIFT` | rise `lift_height` | reached the height or timeout |
| `TRANSPORT` | shift so the object is above the goal | xy error < `pos_tol` or timeout |
| `LOWER` | descend until the object rests on the table | z error < 6 mm or timeout |
| `RELEASE` | open gripper | `release_steps` elapsed |
| `RETREAT` | back away | 25 steps |
| `DONE` | hold, gripper open | — |

   **Recovery:** if during `LIFT/TRANSPORT/LOWER` the grasp is lost (any active arm) for `regrasp_after` steps, it returns to `PRE`.
   In bimanual episodes both hands follow the same phase, so they close, lift, move and release together.
   **Timeouts** (`phase_timeout`) prevent deadlock: a stuck phase gives up and advances.

### API
```python
from sysmic_rl.imitation.expert import ScriptedExpert, ExpertConfig
expert = ScriptedExpert(ExpertConfig())
obs, info = env.reset(seed=0); expert.reset()      # reset at every episode start!
action = expert.act(env)                            # takes the ENV (privileged), returns a (13,) float32 action
```

### `ExpertConfig` reference <a name="tuning-the-expert"></a>

| Field | Default | If you see… | Try |
|---|---|---|---|
| `pre_dist` | 0.07 m | gripper bumps the object while approaching | larger |
| `lift_height` | 0.08 m | object clips other things when carried | larger |
| `pos_tol` | 0.012 m | phases time out instead of finishing | larger (but grasps get sloppier) |
| `axis_tol` | 0.25 rad | stuck in `PRE` | larger, or check `ArmSpec.tool_axis` |
| `damping` | 0.05 | jerky, huge joint jumps near singularities | larger |
| `pos_gain`, `rot_gain` | 0.3 | slow convergence / oscillation | up if slow, down if oscillating |
| `rot_weight` | 0.15 | position OK but wrong orientation (up); position error large because orientation dominates (down) | tune; 1 rad of tilt ≈ `1/rot_weight` m of position error |
| `seed_tol` | 0.06 rad | seed phase never finishes | larger |
| `retreat_dist` | 0.04 m | gripper drags the object when backing away | larger |
| `lift_col_scale` | 0.5 | lift joint wobbling / overused | smaller = uses lift less |
| `close_steps` | 15 | gripper doesn't squeeze enough before lifting | larger |
| `release_steps` | 12 | object dragged away on release | larger |
| `phase_timeout` | 150 | — | the safety net; rarely changed |
| `regrasp_after` | 6 | thrashing between phases | larger |

### Tuning the expert (practical loop)
1. `uv run python -m sysmic_rl.training.evaluate --expert --episodes 5 --render` and **watch** one failure.
2. Note the phase it was in when it failed (add a `print(PHASE_NAMES[expert.phase])` in your own loop).
3. Fix the cause: unreachable target → geometry (`config.py`); wrong sign/offset → `ArmSpec`/gripper angles; contact trouble → `ExpertConfig`.
4. Repeat; measure with 100 episodes. Object-specific failures usually mean grasp style/offset for that object.

> Grasp points live in `config.OBJECT_SPECS` and the pitches in `TaskConfig.*_pitch` (the env's grasp plan), not in `ExpertConfig`.

> The expert relies on the calibrated `ArmSpec` (grasp offset, tool and closing axes) and gripper angles — re-check them with the probe if you change the robot model.

---

## `dataset.py` — demonstration storage

**Why HDF5:** one file, fast sequential reads, per-episode metadata, compressed, no loading everything into RAM unless asked.

### Layout
```
demos.h5
├── ep_00000/
│   ├── obs             (T+1, 103) float32   observation before every step + the final one
│   ├── actions         (T, 13)             what was EXECUTED (may include noise)
│   ├── expert_actions  (T, 13)             what the expert WANTED (clean labels)
│   ├── rewards         (T,)
│   ├── terminated      (T,) bool
│   ├── truncated       (T,) bool
│   └── attrs: success, object, arm
└── ep_00001/ …
```
**Why two action arrays?** BC wants the clean label (`expert_actions`). RL replay needs the action that actually produced the next
observation (`actions`). With noise injection they differ.

### API
* `DemoWriter(path)` — context manager; `.write_episode(obs, actions, expert_actions, rewards, terminated, truncated, **attrs)`.
* `load_demos(path, successful_only=True, max_episodes=None)` → `(dict_of_flat_arrays, n_episodes)` with keys
  `obs, next_obs, actions, expert_actions, rewards, terminated, truncated`.

Inspect a file:
```python
import h5py
with h5py.File("data/demos.h5") as f:
    print(len(f.keys()), f["ep_00000"]["obs"].shape, dict(f["ep_00000"].attrs))
```

---

## `collect.py` — record demonstrations

```bash
# plain expert demos
uv run python -m sysmic_rl.imitation.collect --episodes 300 --noise 0.1 --out data/demos.h5
# DAgger round
uv run python -m sysmic_rl.imitation.collect --episodes 100 --policy runs/bc/bc_sac.zip --beta 0.3 --out data/dagger1.h5
```

| Flag | Meaning |
|---|---|
| `--episodes` | episodes to attempt (not necessarily saved) |
| `--out` | output HDF5 path |
| `--noise` | std of Gaussian noise added to the **executed** action; labels stay clean |
| `--seed` | base seed; episode *i* uses `seed·100003 + i` |
| `--policy` | SB3 SAC zip; enables DAgger mode |
| `--beta` | probability that the **expert** drives each step (1.0 = pure expert; lower = more policy) |
| `--scene-xml` | override scene path |

Behaviour:
* Prints running expert success every 10 episodes and a per-object summary at the end.
* **Plain mode** keeps only *successful* episodes. **DAgger mode** keeps *all* episodes (even failed — the labels are still valid and failures are exactly the states you need).
* The expert runs **in parallel** every step even when the policy drives, so its labels reflect the visited state.

### Why noise?
A perfect expert only visits perfect states. Adding noise to what's executed (but not to the label) shows the network states slightly off the ideal path *together with the correction*, making BC far more robust (this is the DART idea). 0.05–0.2 works well; too much and the expert fails.

---

## `bc.py` — behaviour cloning

```bash
uv run python -m sysmic_rl.imitation.bc --demos data/demos.h5 data/dagger1.h5 --out runs/bc --epochs 100
```

| Flag | Meaning |
|---|---|
| `--demos` | one or more HDF5 files (concatenated; each loaded with `successful_only=True`) |
| `--out` | folder; writes `bc_sac.zip` and TensorBoard logs |
| `--epochs` | passes over the data |
| `--device` | `auto` / `cpu` / `cuda` |

### What `pretrain_actor(model, obs, labels, ...)` does
* Trains **the SAC actor itself** by minimising `MSE(tanh(mean(obs)), expert_action)`. The deterministic SAC action is `tanh(mean)`, so the cloned
  behaviour transfers with no weight copying.
* Adds a small penalty pulling the actor's `log_std` toward −2 (≈ 0.14 std). Without it SAC starts with std ≈ 1 and its random exploration immediately overpowers the clone.
* Holds out 10 % of transitions, keeps the best validation checkpoint, uses Adam + cosine learning-rate decay.
* The critic is **not** trained here (random) — that's handled in `train_sac.py`.

### Reading the output
`val_mse` is the average squared error per action dimension on unseen transitions. Low is good but **not sufficient**: always
evaluate by running the policy (`training/evaluate.py`), because small per-step errors compound over an episode.

### Common BC issues
| Problem | Cause | Fix |
|---|---|---|
| val loss plateaus high | expert actions inconsistent (e.g. phase thrashing) or too little data | improve the expert; more demos |
| val low but rollouts fail | distribution shift | DAgger, more `--noise` |
| train loss ≪ val loss | overfitting | more data, `weight_decay`, fewer epochs |
| grasps then drops | label for "keep gripper closed" is subtle | more demos; check `close_steps` |

---

## Making changes safely

* **Bring your own demos (teleoperation):** record episodes yourself and write them with `DemoWriter.write_episode(...)`. Observations must come
  from `env._get_obs()`, actions must be in the env's 13-D normalised space, and set `success=True` in attrs. Pass `expert_actions=actions`.
* **Replace the expert** with a different controller: keep the `reset()` / `act(env) -> (13,)` interface and everything else works.
* **Change the BC network:** don't edit `bc.py`; change `net` in `training/common.py:build_sac` (BC trains whichever actor it is given).
* **Add a grasp style:** extend `SysmicPickPlaceEnv.grasp_plan()` (styles: `rim`, `stem`, `handle`).
* After changing the expert always re-run `evaluate --expert`, then **re-collect** — old demos reflect the old expert.
