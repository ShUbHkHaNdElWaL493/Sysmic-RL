"""Configuration for the Sysmic pick-and-place task.

Every number here that depends on the *real* robot geometry (grasp offsets,
gripper open/close angles, table pose, spawn region) is a first guess derived
from the MJCF. Calibrate them with `python -m sysmic_rl.pickplace.probe`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Override with SYSMIC_ROBOT_XML=/abs/path/to/robot.xml if your file is named differently.
DEFAULT_ROBOT_XML = os.environ.get("SYSMIC_ROBOT_XML", str(REPO_ROOT / "mjcf" / "sysmic_v2.xml"))


@dataclass(frozen=True)
class ArmSpec:
    side: str
    joints: tuple          # pan, shoulder, elbow, wrist_flex, wrist_roll, gripper
    finger_body: str       # body carrying the fixed finger / palm (wrist roll link)
    jaw_body: str          # moving jaw
    grasp_offset: tuple    # grasp point in finger_body frame (default: jaw COM, CALIBRATE)
    tool_axis: tuple = (1.0, 0.0, 0.0)  # approach axis in finger_body frame


LEFT_ARM = ArmSpec(
    "left",
    (
        "arm_left_pan_joint",
        "arm_left_shoulder_joint",
        "arm_left_elbow_joint",
        "arm_left_wrist_flex_joint",
        "arm_left_wrist_roll_joint",
        "arm_left_gripper_joint"
    ),
    "arm_left_wrist_roll_link",
    "arm_left_jaw_link",
    (0.0576, 0.0236, 0.0436),
)
RIGHT_ARM = ArmSpec(
    "right",
    (
        "arm_right_pan_joint",
        "arm_right_shoulder_joint",
        "arm_right_elbow_joint",
        "arm_right_wrist_flex_joint",
        "arm_right_wrist_roll_joint",
        "arm_right_gripper_joint"
    ),
    "arm_right_wrist_roll_link",
    "arm_right_jaw_link",
    (0.0576, -0.0259, 0.0423),
)
ARMS = {"left": LEFT_ARM, "right": RIGHT_ARM}


@dataclass
class SceneConfig:
    robot_xml: str = DEFAULT_ROBOT_XML
    objects: tuple = ("plate", "glass", "bowl", "mug")
    object_scale: float = 1.0
    friction: tuple = (1.0, 0.005, 0.0001)
    # Table (world frame == robot base frame; the robot base is fixed at the origin)
    table_top_z: float = 0.72
    table_center_xy: tuple = (0.32, 0.0)
    table_half_xy: tuple = (0.20, 0.45)
    table_thickness: float = 0.04
    # Object spawn / goal region on the table (x, y ranges)
    spawn_x: tuple = (0.22, 0.34)
    spawn_y: tuple = (-0.25, 0.25)


@dataclass
class TaskConfig:
    arm_mode: str = "auto"            # "left" | "right" | "auto" (random per episode)
    max_episode_steps: int = 400
    frame_skip: int = 10              # 10 * 2 ms = 50 Hz policy
    ema_alpha: float = 0.25
    arm_action_scale: float = 0.03    # rad / step
    gripper_action_scale: float = 0.08
    lift_action_scale: float = 0.002  # m / step
    arm_max_lag: float = 0.15
    gripper_max_lag: float = 0.30
    lift_max_lag: float = 0.03
    # CALIBRATE: which gripper angle is open/closed (probe prints a sweep)
    gripper_open_q: float = 1.2
    gripper_close_q: float = -0.1
    arm_home: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 1.2)
    home_noise: float = 0.05
    lift_home_range: tuple = (0.05, 0.30)
    min_goal_dist: float = 0.10
    # success
    place_tol_xy: float = 0.02
    place_tol_z: float = 0.015
    place_max_tilt: float = 0.35
    place_max_speed: float = 0.05
    hold_steps: int = 10
    fail_tilt: float = 1.1            # rad, terminate if object tips over this far
    lift_target: float = 0.08
    # domain randomisation
    randomize_physics: bool = True
    friction_scale: tuple = (0.7, 1.3)
    mass_scale: tuple = (0.8, 1.2)


@dataclass
class RewardConfig:
    reach: float = 1.0
    reach_scale: float = 8.0
    grasp: float = 1.0
    lift: float = 2.0
    goal: float = 3.0
    goal_scale: float = 6.0
    place: float = 2.0
    release: float = 1.0
    success: float = 50.0
    collision: float = 0.5
    collision_pen: float = 20.0
    collision_cap: float = 3.0
    action_rate: float = 0.002
    velocity: float = 0.0005
    fail: float = 10.0


@dataclass
class PickPlaceConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self):
        return asdict(self)
