"""Collect demonstrations with the scripted expert, optionally DAgger-style.

  uv run python -m sysmic_rl.imitation.collect --episodes 300 --noise 0.1 --out data/demos.h5
  # DAgger round (policy drives with prob. 1-beta, expert labels every state):
  uv run python -m sysmic_rl.imitation.collect --episodes 100 --policy runs/bc/bc_sac.zip --beta 0.3 --out data/dagger1.h5
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from sysmic_rl.imitation.dataset import DemoWriter
from sysmic_rl.imitation.expert import ExpertConfig, ScriptedExpert
from sysmic_rl.env import SysmicBotConfig, SysmicBotEnv


def collect(n_episodes, out, cfg=None, noise=0.0, seed=0, policy=None, beta=1.0, keep_failures=None, log=print):
    cfg = cfg or SysmicBotConfig()
    env = SysmicBotEnv(cfg)
    expert = ScriptedExpert(ExpertConfig())
    rng = np.random.default_rng(seed)
    keep_failures = (policy is not None) if keep_failures is None else keep_failures
    stats, per_obj = Counter(), Counter()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with DemoWriter(out) as w:
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed * 100003 + ep)
            expert.reset()
            O, A, E, R, T, U = [obs], [], [], [], [], []
            done = False
            while not done:
                a_exp = expert.act(env)
                if policy is not None and rng.random() > beta:
                    a_exec = policy.predict(obs, deterministic=True)[0]
                else:
                    a_exec = np.clip(a_exp + noise * rng.standard_normal(13), -1, 1) * env.action_mask
                obs, r, term, trunc, step_info = env.step(a_exec)
                O.append(obs); A.append(a_exec); E.append(a_exp); R.append(r); T.append(term); U.append(trunc)
                done = term or trunc
            ok = bool(step_info["success"])
            stats["success" if ok else "fail"] += 1
            per_obj[(info["object"], ok)] += 1
            if ok or keep_failures:
                w.write_episode(O, A, E, R, T, U, success=ok, object=info["object"], arm=info["arm"])
            if (ep + 1) % 10 == 0:
                log(f"[{ep + 1}/{n_episodes}] expert success rate {stats['success'] / (ep + 1):.2f}")
    env.close()
    log(f"done: {dict(stats)}")
    for n in cfg.scene.objects:
        s, f = per_obj[(n, True)], per_obj[(n, False)]
        log(f"  {n:6s}: {s}/{s + f} success")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="data/demos.h5")
    p.add_argument("--noise", type=float, default=0.1, help="exec-noise std (labels stay clean)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", default=None, help="SB3 SAC zip for DAgger rollouts")
    p.add_argument("--beta", type=float, default=1.0, help="prob. expert drives each step (DAgger mix)")
    p.add_argument("--scene-xml", default=None)
    a = p.parse_args()
    cfg = SysmicBotConfig()
    if a.scene_xml:
        cfg.scene.scene_xml = a.scene_xml
    pol = None
    if a.policy:
        from stable_baselines3 import SAC
        pol = SAC.load(a.policy, device="cpu")
    collect(a.episodes, a.out, cfg, a.noise, a.seed, pol, a.beta)
