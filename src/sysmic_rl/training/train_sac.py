"""SAC fine-tuning initialised from demonstrations.

  1. (optional) load BC-pretrained actor
  2. pre-fill the replay buffer with demo transitions (sparse-ish reward is already in the demos)
  3. critic warm-up with the actor frozen (so the BC actor is not destroyed by a random critic)
  4. online SAC

  uv run python -m sysmic_rl.training.train_sac --demos data/demos.h5 --init runs/bc/bc_sac.zip --out runs/sac
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sysmic_rl.imitation.dataset import load_demos
from sysmic_rl.pickplace import PickPlaceConfig
from sysmic_rl.training.common import build_sac, make_vec_env


def prefill_buffer(model, demos):
    buf = model.replay_buffer
    n = len(demos["obs"])
    for i in range(n):
        trunc = bool(demos["truncated"][i] and not demos["terminated"][i])
        buf.add(demos["obs"][i][None], demos["next_obs"][i][None], demos["actions"][i][None],
                np.array([demos["rewards"][i]], dtype=np.float32),
                np.array([demos["terminated"][i] or demos["truncated"][i]]),
                [{"TimeLimit.truncated": trunc}])
    return n


def critic_warmup(model, steps, batch_size=256):
    for p in model.actor.parameters():
        p.requires_grad_(False)
    model.train(gradient_steps=steps, batch_size=batch_size)
    for p in model.actor.parameters():
        p.requires_grad_(True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="*", default=[])
    ap.add_argument("--init", default=None, help="BC checkpoint (SAC zip) to start from")
    ap.add_argument("--out", default="runs/sac")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=20_000, help="critic-only gradient steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--robot-xml", default=None)
    a = ap.parse_args()

    cfg = PickPlaceConfig()
    if a.robot_xml:
        cfg.scene.robot_xml = a.robot_xml
    env, eval_env = make_vec_env(cfg, 0), make_vec_env(cfg, 10_000)
    model = build_sac(env, a.out, device=a.device)
    if a.init:
        model.set_parameters(a.init, exact_match=False, device=model.device)  # actor weights (critic left fresh)
        print(f"initialised from {a.init}")
    if a.demos:
        n = sum(prefill_buffer(model, load_demos(f, successful_only=False)[0]) for f in a.demos)
        print(f"replay buffer pre-filled with {n} demo transitions")
        if a.warmup:
            critic_warmup(model, a.warmup)
    cbs = [CheckpointCallback(100_000, a.out, name_prefix="sac"),
           EvalCallback(eval_env, best_model_save_path=a.out, log_path=a.out, eval_freq=20_000,
                        n_eval_episodes=20, deterministic=True)]
    model.learn(a.steps, callback=cbs, reset_num_timesteps=True, progress_bar=False)
    model.save(f"{a.out}/final.zip")
