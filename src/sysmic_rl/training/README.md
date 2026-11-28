# `sysmic_rl.training` — the learner

This package contains the reinforcement-learning side: creating the SAC agent, fine-tuning it starting from demonstrations,
and measuring how good any policy is. It depends on `stable-baselines3` (SB3) and PyTorch.

```
training/
├── common.py      make_vec_env() and build_sac(): the shared setup
├── train_sac.py   demo-seeded SAC fine-tuning (the long-running script)
└── evaluate.py    success rate for the expert, a BC model, or an RL model
```

## SAC in five sentences

**SAC (Soft Actor-Critic)** is an *off-policy* RL algorithm for continuous actions. It trains an **actor** (the policy) and two **critics**
(estimates of future reward for a given obs+action). The actor is updated to pick actions the critics like, *plus* an entropy bonus that
keeps it exploring. Because it learns from a **replay buffer** of stored transitions, it can reuse old experience — including expert
demonstrations we put in on purpose. The temperature `ent_coef` (how much it values randomness) is adjusted automatically.

---

## `common.py`

### `make_vec_env(cfg=None, seed=0)`
Builds a `DummyVecEnv` with one `Monitor`-wrapped `SysmicPickPlaceEnv`.
* **Why `Monitor`:** records episode return/length (and `is_success`) for logging.
* **Why a vector env with 1 copy:** SB3 requires the VecEnv interface. The demo pre-fill in `train_sac.py` assumes exactly **one** env; for several you'd change `prefill_buffer`.
* Note: normalisation is **inside** the env, so no `VecNormalize` — adding it would silently desynchronise BC and RL.

### `build_sac(env, log_dir, device="auto", ent_init=0.01, lr=3e-4, net=(256,256))`
Creates the SAC model with these settings (edit here to change them):

| Setting | Value | Meaning / when to change |
|---|---|---|
| `learning_rate` | 3e-4 | lower (1e-4) if training is unstable |
| `buffer_size` | 1,000,000 | replay capacity; must exceed demos + online data you want to keep |
| `learning_starts` | 0 | start learning at once (the buffer is pre-filled with demos) |
| `batch_size` | 256 | |
| `tau` | 0.005 | how slowly target critics follow |
| `gamma` | 0.99 | discount: how far ahead it cares. Episodes are 400 steps, 0.99 ≈ 100-step horizon |
| `train_freq`, `gradient_steps` | 1, 1 | one update per environment step |
| `ent_coef` | `"auto_0.01"` | auto-tuned, starting small so the BC policy isn't drowned in noise |
| `net_arch` | pi/qf = (256, 256) | **BC checkpoints only load into the same size** |

It also installs a logger (`stdout` + TensorBoard in `log_dir`). Setting it explicitly stops SB3 from replacing it, so BC and warm-up
metrics end up in the same run.

---

## `train_sac.py` — demo-seeded fine-tuning

```bash
uv run python -m sysmic_rl.training.train_sac \
   --demos data/demos.h5 data/dagger1.h5 --init runs/bc2/bc_sac.zip --out runs/sac --steps 1000000 --warmup 20000
```

| Flag | Default | Meaning |
|---|---|---|
| `--demos` | none | demo files for the replay buffer (all episodes, incl. failures, are used) |
| `--init` | none | BC checkpoint to start the actor from (without it: RL from scratch) |
| `--out` | `runs/sac` | logs/checkpoints folder |
| `--steps` | 1,000,000 | environment steps of online training |
| `--warmup` | 20,000 | critic-only gradient steps before the actor may change (0 = skip) |
| `--device` | auto | `cpu`/`cuda` |
| `--scene-xml` | – | scene path override |

### Order of operations (and why)
1. **Build** the agent and two envs (train + eval, different seeds).
2. **Load `--init`** (`set_parameters(..., exact_match=False)`): the actor starts as the BC policy.
3. **`prefill_buffer`**: add every demo transition to the replay buffer with its stored reward, the *executed* action, and a correct
   `done`/"time-limit truncated" flag. *Why:* the critic can immediately learn what "good" looks like (reward-bearing states) instead of seeing only the clumsy early attempts.
4. **`critic_warmup`**: train for `--warmup` steps with **actor parameters frozen**. *Why:* a freshly initialised critic gives garbage gradients; if the actor
   followed them from step 0 it would forget the BC behaviour in a few hundred updates. Freezing lets the critic catch up first.
5. **`model.learn`**: standard SAC, with callbacks:
   * `CheckpointCallback`: save every 100k steps as `sac_<n>_steps.zip`.
   * `EvalCallback`: every 20k steps run 20 deterministic eval episodes; log `eval/mean_reward` and `eval/success_rate`; keep `best_model.zip`.
6. Save `final.zip`.

### Outputs
`runs/sac/` → TensorBoard event files, `best_model.zip`, `final.zip`, `sac_*_steps.zip`, `evaluations.npz`.
Resume ideas: load a checkpoint with `SAC.load(path, env=env)` and call `learn(..., reset_num_timesteps=False)` (not wired into the CLI).

### Troubleshooting <a name="troubleshooting"></a>

| Symptom | Likely reason | What to try |
|---|---|---|
| Success drops to ~0 right after start | critic not ready / too much entropy | raise `--warmup` to 50k–100k; `ent_init` 0.001; `lr` 1e-4 |
| Never improves beyond BC | reward doesn't distinguish better from worse; or too little exploration | inspect `info["reward_terms"]`; try `ent_init` 0.05 |
| Reward goes up, success doesn't | reward hacking | rebalance `RewardConfig` (e.g. lower `reach`, raise `place/success`) — then **re-collect demos** |
| `critic_loss` explodes | reward scale too large | scale rewards down; success bonus 50 is already the max |
| Very slow | 1 env, MuJoCo on CPU | accept (≈1–3k steps/s) or build a vectorised version |
| `AssertionError`/shape errors in `prefill_buffer` | using more than 1 env, or obs size differs from demos | single env; re-collect demos |
| `set_parameters` mismatch | network size/obs size changed | retrain BC with the same `net` |

---

## `evaluate.py` — measure success

```bash
uv run python -m sysmic_rl.training.evaluate --expert --episodes 100
uv run python -m sysmic_rl.training.evaluate --model runs/bc/bc_sac.zip --episodes 100
uv run python -m sysmic_rl.training.evaluate --model runs/sac/best_model.zip --episodes 100 --render
```

* Runs fixed seeds (1000+i) so different models are compared on the same episodes.
* Deterministic policy (no exploration noise) — `predict(obs, deterministic=True)`.
* Prints success rate overall, per object, per arm mode (`arm:left`, `arm:right`, `arm:both` = bimanual episodes) and mean return.
* Python API: `evaluate(policy_fn, cfg, episodes, seed)` where `policy_fn(obs, env) -> action`; give it a `.reset()` method if it has state (the expert does).

**Always judge progress by success rate, not return** — return can rise from partial progress or reward exploitation.

---

## Beginner guide: how much should I expect?

* Expert demos + BC: usually works in minutes of training; sets the floor.
* DAgger: a few rounds close a good part of the BC-vs-expert gap.
* RL fine-tuning: hours; gains are typically gradual. Compare against the BC-only baseline using the same `evaluate` seeds.
* A clean experiment table (BC-only / BC+RL / RL-from-scratch) is the best way to understand what each piece contributes — run `train_sac` without `--demos`/`--init` for the scratch baseline.

## Making changes safely

| Want to… | Do this |
|---|---|
| Try PPO instead of SAC | build `PPO("MlpPolicy", env, ...)` in a new script. BC pretraining works differently (PPO's actor is a Gaussian, no `tanh`), and the buffer pre-fill/critic warm-up steps don't apply — start by copying `build_sac` and removing those |
| Train on a GPU | `--device cuda` (small MLPs are often faster on CPU) |
| More logging | add a `BaseCallback` subclass and pass it in `cbs` in `train_sac.py` |
| Curriculum (easier task first) | train with a restricted `PickPlaceConfig` (e.g. `scene.objects=("glass",)`, `arm_mode="left"`) — note object list changes obs size |
| Save more checkpoints | change the `CheckpointCallback` frequency |
| Multi-env speed-up | replace `DummyVecEnv` with `SubprocVecEnv`, and rewrite `prefill_buffer` to add rows of shape `(n_envs, …)` |

Reminder: after any edit to the env (obs, reward, actions) **demos and checkpoints are stale** — see the compatibility rules in
[README §7](../../../README.md#7-i-want-to-change-x--where-do-i-go).
