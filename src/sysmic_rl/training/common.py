from __future__ import annotations

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sysmic_rl.env import SysmicBotConfig, SysmicBotEnv


def make_vec_env(cfg: SysmicBotConfig | None = None, seed: int = 0):
    """Single-env VecEnv. (Demo pre-filling assumes n_envs == 1.)"""
    def _thunk():
        env = Monitor(SysmicBotEnv(cfg or SysmicBotConfig()), info_keywords=("is_success",))
        env.reset(seed=seed)
        return env
    return DummyVecEnv([_thunk])


def build_sac(env, log_dir: str, device="auto", ent_init=0.01, lr=3e-4, net=(256, 256)) -> SAC:
    model = SAC(
        "MlpPolicy", env, learning_rate=lr, buffer_size=1_000_000, learning_starts=0, batch_size=256,
        tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef=f"auto_{ent_init}",
        policy_kwargs=dict(net_arch=dict(pi=list(net), qf=list(net))), device=device, verbose=1,
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # custom logger => SB3 keeps it inside learn(), so BC / warm-up metrics land in the same TensorBoard run
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
    return model
