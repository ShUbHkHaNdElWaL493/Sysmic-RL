"""HDF5 demonstration storage: one group per episode."""
from __future__ import annotations

import h5py
import numpy as np


class DemoWriter:
    def __init__(self, path, mode="w"):
        self.f = h5py.File(path, mode)
        self.n = len(self.f.keys())

    def write_episode(self, obs, actions, expert_actions, rewards, terminated, truncated, **attrs):
        g = self.f.create_group(f"ep_{self.n:05d}")
        kw = dict(compression="lzf")
        g.create_dataset("obs", data=np.asarray(obs, np.float32), **kw)  # (T+1, D)
        g.create_dataset("actions", data=np.asarray(actions, np.float32), **kw)  # executed
        g.create_dataset("expert_actions", data=np.asarray(expert_actions, np.float32), **kw)  # labels
        g.create_dataset("rewards", data=np.asarray(rewards, np.float32))
        g.create_dataset("terminated", data=np.asarray(terminated, bool))
        g.create_dataset("truncated", data=np.asarray(truncated, bool))
        for k, v in attrs.items():
            g.attrs[k] = v
        self.n += 1
        self.f.flush()

    def close(self): self.f.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def load_demos(path, successful_only=True, max_episodes=None):
    """Return flat transition arrays (obs, next_obs, actions, expert_actions, rewards, terminated, truncated)."""
    out = {k: [] for k in ("obs", "next_obs", "actions", "expert_actions", "rewards", "terminated", "truncated")}
    used = 0
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys()):
            g = f[name]
            if successful_only and not g.attrs.get("success", False):
                continue
            o = g["obs"][:]
            out["obs"].append(o[:-1]); out["next_obs"].append(o[1:])
            for k in ("actions", "expert_actions", "rewards", "terminated", "truncated"):
                out[k].append(g[k][:])
            used += 1
            if max_episodes and used >= max_episodes:
                break
    if not used:
        raise ValueError(f"No usable episodes in {path} (successful_only={successful_only})")
    return {k: np.concatenate(v) for k, v in out.items()}, used
