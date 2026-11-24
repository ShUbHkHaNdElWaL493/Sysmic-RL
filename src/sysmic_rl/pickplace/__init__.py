import gymnasium as gym

from .config import ARMS, PickPlaceConfig, RewardConfig, SceneConfig, TaskConfig
from .pickplace_env import SysmicPickPlaceEnv

if "SysmicPickPlace-v0" not in gym.registry:
    gym.register(id="SysmicPickPlace-v0", entry_point="sysmic_rl.pickplace.pickplace_env:SysmicPickPlaceEnv")

__all__ = ["SysmicPickPlaceEnv", "PickPlaceConfig", "SceneConfig", "TaskConfig", "RewardConfig", "ARMS"]
