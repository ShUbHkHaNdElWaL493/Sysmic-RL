"""Configuration for the Sysmic pick-and-place task.

Every number here that depends on the *real* robot geometry (grasp offsets,
gripper open/close angles, table pose, spawn region) is a first guess derived
from the MJCF. Calibrate them with `python -m sysmic_rl.pickplace.probe`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[3]
ROBOT_XML = os.environ.get("ROBOT_XML", str(WS_ROOT / "mjcf" / "sysmic_v2.xml"))


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
    ("arm_left_pan_joint", "arm_left_shoulder_joint", "arm_left_elbow_joint",
     "arm_left_wrist_flex_joint", "arm_left_wrist_roll_joint", "arm_left_gripper_joint"),
    "arm_left_wrist_roll_link", "arm_left_jaw_link", (0.0576, 0.0236, 0.0436),
)
RIGHT_ARM = ArmSpec(
    "right",
    ("arm_right_pan_joint", "arm_right_shoulder_joint", "arm_right_elbow_joint",
     "arm_right_wrist_flex_joint", "arm_right_wrist_roll_joint", "arm_right_gripper_joint"),
    "arm_right_wrist_roll_link", "arm_right_jaw_link", (0.0576, -0.0259, 0.0423),
)
ARMS = {"left": LEFT_ARM, "right": RIGHT_ARM}


@dataclass
class SceneConfig:
    robot_xml: str = ROBOT_XML
    objects: tuple = ("plate", "glass", "bowl", "mug")
    object_scale: float = 1.0
    friction: tuple = (1.0, 0.005, 0.0001)
    # --- destination table: spawned every episode (centre x/y ranges, top-surface height range)
    table_half_xy: tuple = (0.18, 0.22)
    table_thickness: float = 0.04
    table_x: tuple = (0.25, 0.55)
    table_y: tuple = (-0.25, 0.25)
    table_z: tuple = (0.55, 1.05)
    table_clearance: float = 0.03      # min gap between table and any robot geom
    # --- reachability filtering (kinematic reach cloud; see pickplace_env._build_reach_cloud)
    reach_check: bool = True
    reach_samples: int = 50000
    reach_tol: float = 0.04            # grasp point must be within this of a reachable sample
    reach_axis_tol: float = 0.5        # rad: sample's tool axis must be this close to the needed one


@dataclass
class SourceConfig:
    """Where objects START: a flat surface that belongs to the robot (the 'extrusion').

    manual=None -> auto-detected by casting rays downwards onto the robot's collision geoms.
    manual=(cx, cy, z, hx, hy) -> world-frame rectangle at the home pose (use if detection picks the wrong patch).
    body -> body the surface is attached to (moves with it). None = auto (detected) / world (manual).
    """
    manual: tuple | None = None
    body: str | None = None
    search_x: tuple = (-0.10, 0.60)
    search_y: tuple = (-0.45, 0.45)
    search_z: tuple = (0.30, 1.40)
    step: float = 0.01
    min_area: float = 0.01             # m^2: smallest patch accepted as "the surface"
    edge_margin: float = 0.015         # keep objects this far from the patch edge


@dataclass
class TaskConfig:
    arm_mode: str = "auto"            # "left" | "right" | "auto" (random per episode)
    max_episode_steps: int = 500
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
    # --- grasp geometry (shared by env reward/reachability and the expert)
    side_height_frac: float = 0.6      # "side" grasp height as a fraction of object height
    top_depth: float = 0.006           # "top" grasp: how far below the rim top the grasp point sits
    # --- bimanual episodes
    bimanual_objects: tuple = ("plate", "bowl")
    bimanual_prob: float = 1.0         # prob. an object in bimanual_objects is carried with BOTH arms
    bimanual_y_max: float = 0.10       # |y| limit of the object so both arms can reach its two sides
    bimanual_style: str = "down"       # "down": both tools point down onto opposite rims; "inward": tools point at each other
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
    source: SourceConfig = field(default_factory=SourceConfig)

    def to_dict(self):
        return asdict(self)
