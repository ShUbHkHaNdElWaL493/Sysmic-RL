import gymnasium as gym

from .config import ARMS, SysmicBotConfig, RewardConfig, SceneConfig, TaskConfig
from .env import SysmicBotEnv

if "SysmicBot-v1" not in gym.registry:
    gym.register(id="SysmicBot-v1", entry_point="sysmic_rl.env.env:SysmicBotEnv")

__all__ = ["SysmicBotEnv", "SysmicBotConfig", "SceneConfig", "TaskConfig", "RewardConfig", "ARMS"]
