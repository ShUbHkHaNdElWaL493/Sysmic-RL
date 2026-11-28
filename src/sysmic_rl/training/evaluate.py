"""Success-rate evaluation (per object, per arm).  Works for BC, RL, or the scripted expert (--expert)."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from sysmic_rl.pickplace import PickPlaceConfig, SysmicPickPlaceEnv


def evaluate(policy_fn, cfg=None, episodes=100, seed=1000, render=False):
    env = SysmicPickPlaceEnv(cfg or PickPlaceConfig(), render_mode="human" if render else None)
    per = defaultdict(list)
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        done, R = False, 0.0
        while not done:
            obs, r, te, tr, si = env.step(policy_fn(obs, env))
            R += r; done = te or tr
        per[info["object"]].append(si["success"]); per[f"arm:{info['arm']}"].append(si["success"])
        per["ALL"].append(si["success"]); returns.append(R)
    env.close()
    for k, v in sorted(per.items()):
        print(f"{k:12s} success {np.mean(v):.2f}  ({int(np.sum(v))}/{len(v)})")
    print(f"mean return {np.mean(returns):.1f}")
    return float(np.mean(per["ALL"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model"); p.add_argument("--expert", action="store_true")
    p.add_argument("--episodes", type=int, default=100); p.add_argument("--render", action="store_true")
    p.add_argument("--scene-xml", default=None)
    a = p.parse_args()
    cfg = PickPlaceConfig()
    if a.scene_xml:
        cfg.scene.scene_xml = a.scene_xml
    if a.expert:
        from sysmic_rl.imitation.expert import ScriptedExpert
        ex = ScriptedExpert()
        class Pol:
            def reset(self): ex.reset()
            def __call__(self, obs, env): return ex.act(env)
        pol = Pol()
    else:
        from stable_baselines3 import SAC
        m = SAC.load(a.model, device="cpu")
        pol = lambda obs, env: m.predict(obs, deterministic=True)[0]
    evaluate(pol, cfg, a.episodes, render=a.render)
