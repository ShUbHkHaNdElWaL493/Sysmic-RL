"""Behaviour-cloning pre-training of an SB3 SAC actor.

The BC policy *is* the SAC actor (deterministic head tanh(mu)), so no weight copying is needed
and RL fine-tuning starts from the cloned behaviour.

  uv run python -m sysmic_rl.imitation.bc --demos data/demos.h5 --out runs/bc
"""
from __future__ import annotations

import argparse

import numpy as np
import torch as th

from sysmic_rl.imitation.dataset import load_demos


def pretrain_actor(model, obs, labels, epochs=100, batch_size=512, lr=3e-4, val_frac=0.1,
                   target_log_std=-2.0, std_weight=0.1, seed=0, log=print):
    """MSE on the deterministic action + pull log-std down so early RL exploration doesn't wreck the clone."""
    actor, dev = model.actor, model.device
    n = len(obs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    va, tr = perm[:nv], perm[nv:]
    X = th.as_tensor(obs, dtype=th.float32, device=dev)
    Y = th.as_tensor(labels, dtype=th.float32, device=dev)
    opt = th.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-5)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = (1e9, None)
    for ep in range(epochs):
        actor.train()
        np.random.default_rng(seed + ep).shuffle(tr)
        for i in range(0, len(tr), batch_size):
            idx = th.as_tensor(tr[i:i + batch_size], device=dev)
            mean, log_std, _ = actor.get_action_dist_params(X[idx])
            pred = th.tanh(mean)
            loss = th.nn.functional.mse_loss(pred, Y[idx]) + std_weight * ((log_std - target_log_std) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with th.no_grad():
            actor.eval()
            vl = th.nn.functional.mse_loss(actor(X[th.as_tensor(va, device=dev)], deterministic=True), Y[th.as_tensor(va, device=dev)]).item()
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in actor.state_dict().items()})
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"[bc] epoch {ep:3d}  val_mse {vl:.5f}  (best {best[0]:.5f})")
    actor.load_state_dict(best[1])
    actor.eval()
    return {"val_mse": best[0]}


if __name__ == "__main__":
    from sysmic_rl.pickplace import PickPlaceConfig
    from sysmic_rl.training.common import build_sac, make_vec_env

    p = argparse.ArgumentParser()
    p.add_argument("--demos", nargs="+", required=True)
    p.add_argument("--out", default="runs/bc")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--scene-xml", default=None)
    a = p.parse_args()
    cfg = PickPlaceConfig()
    if a.scene_xml:
        cfg.scene.scene_xml = a.scene_xml
    env = make_vec_env(cfg)
    model = build_sac(env, a.out, device=a.device)
    parts = [load_demos(f)[0] for f in a.demos]
    obs = np.concatenate([d["obs"] for d in parts])
    lab = np.concatenate([d["expert_actions"] for d in parts])
    print(f"BC on {len(obs)} transitions")
    pretrain_actor(model, obs, lab, epochs=a.epochs)
    model.save(f"{a.out}/bc_sac.zip")
    print(f"saved {a.out}/bc_sac.zip  -> evaluate with: python -m sysmic_rl.training.evaluate --model {a.out}/bc_sac.zip")
