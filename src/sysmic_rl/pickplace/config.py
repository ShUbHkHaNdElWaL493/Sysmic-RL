"""Configuration for the Sysmic pick-and-place task.

Every number here that depends on the *real* robot geometry (grasp offsets,
gripper open/close angles, table pose) is a first guess derived from the MJCF.
Calibrate them with `python -m sysmic_rl.pickplace.probe`.
The scene itself (robot + shelf + table + utensils) is the static `mjcf/sysmic_v2_scene.xml`.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[3]
SCENE_XML = os.environ.get("SYSMIC_SCENE_XML", str(WS_ROOT / "mjcf" / "sysmic_v2_scene.xml"))


@dataclass(frozen=True)
class ObjectSpec:
    """Metadata for a utensil defined in the scene XML (body `obj_<name>`, origin = centre of the base bottom, +z up)."""
    name: str
    height: float
    radius: float        # largest radius of the body around its axis (handle excluded)
    extent: float        # largest horizontal distance from the axis (handle included)
    grasp_style: str     # "rim" (pinch the rim wall) | "stem" (close on the stem) | "handle" (close on the handle's outer bar)
    grasp_r: float       # horizontal distance axis -> grasp point (rim wall / handle bar; 0 for a stem on the axis)
    grasp_z: float       # height of the grasp point above the base
    wall_slope: float    # "rim": wall angle from horizontal at the grasp point (the jaws close across the wall)
    footprint_y: float   # half-width along the shelf (y) when spawned


# Must match the geometry in mjcf/sysmic_v2_scene.xml (see the comment above each utensil body there).
OBJECT_SPECS = {
    #                         height  radius extent  style     grasp_r grasp_z slope  footprint_y
    "plate": ObjectSpec("plate", 0.022, 0.062, 0.062, "rim", 0.059, 0.0187, 0.527, 0.062),
    "bowl": ObjectSpec("bowl", 0.050, 0.065, 0.065, "rim", 0.0626, 0.042, 1.355, 0.065),
    "glass": ObjectSpec("glass", 0.140, 0.035, 0.035, "stem", 0.0, 0.035, 0.0, 0.035),
    "mug": ObjectSpec("mug", 0.085, 0.036, 0.065, "handle", 0.060, 0.0475, 0.0, 0.036),   # handle on local +x
}


@dataclass(frozen=True)
class ArmSpec:
    side: str
    joints: tuple          # pan, shoulder, elbow, wrist_flex, wrist_roll, gripper
    finger_body: str       # body carrying the fixed finger (wrist roll link)
    jaw_body: str          # moving jaw
    grasp_offset: tuple    # grasp point in finger_body frame: where a ~1 cm object sits when clamped
    tool_axis: tuple       # approach axis in finger_body frame (along the fingers, = wrist roll axis)
    close_axis: tuple      # direction the jaw moves when closing, in finger_body frame (perpendicular to tool_axis)


# Calibrated from the MJCF (meshes collide as convex hulls): a 1 cm rod placed 3 cm behind the fingertip rests
# against the fixed finger's hull at grasp_offset, and the jaw reaches it at q ~ 1.5 (< gripper_close_q, so it
# squeezes). The grasp point is ~6 mm from the wrist-roll axis, so rolling the wrist turns the closing direction
# without moving the grasp point. Re-check with the probe's "gripper" section.
LEFT_ARM = ArmSpec(
    "left",
    ("arm_left_pan_joint", "arm_left_shoulder_joint", "arm_left_elbow_joint",
     "arm_left_wrist_flex_joint", "arm_left_wrist_roll_joint", "arm_left_gripper_joint"),
    "arm_left_wrist_roll_link", "arm_left_jaw_link",
    (0.120, 0.0027, 0.0047), (1.0, 0.0, 0.0), (0.0, -0.5, -0.866),
)
RIGHT_ARM = ArmSpec(
    "right",
    ("arm_right_pan_joint", "arm_right_shoulder_joint", "arm_right_elbow_joint",
     "arm_right_wrist_flex_joint", "arm_right_wrist_roll_joint", "arm_right_gripper_joint"),
    "arm_right_wrist_roll_link", "arm_right_jaw_link",
    (0.120, -0.0027, 0.0047), (1.0, 0.0, 0.0), (0.0, 0.5, -0.866),
)
ARMS = {"left": LEFT_ARM, "right": RIGHT_ARM}


@dataclass
class SceneConfig:
    scene_xml: str = SCENE_XML
    objects: tuple = ("plate", "glass", "bowl", "mug")   # all must exist in the scene XML as obj_<name>
    # --- source: every utensil starts on the shelf riser (site in the scene XML), in a shuffled row along y
    shelf_site: str = "shelf"
    shelf_min_gap: float = 0.01        # min free space between neighbouring utensils
    # --- destination table, in front of the robot: x/y/yaw re-sampled every episode, size from the XML.
    table_z: float = 0.85              # constant top-surface height (must match the table body in the XML)
    table_gap: tuple = (0.0, 0.08)     # distance of the table's nearest edge beyond the front limit
                                       # (front limit = front edge of the shelf + table_clearance)
    table_y: tuple = (-0.15, 0.15)     # table centre y
    table_yaw: tuple = (-math.pi / 2, math.pi / 2)   # covers every distinct orientation of a rectangle
    table_clearance: float = 0.03      # min gap between table and any robot geom / the shelf
    goal_margin: float = 0.01          # min distance between the placed object's footprint and the table edge
    # --- reachability filtering (kinematic reach cloud; see pickplace_env._build_reach_cloud)
    reach_check: bool = True
    reach_samples: int = 50000
    reach_tol: float = 0.04            # grasp point must be within this of a reachable sample
    reach_axis_tol: float = 0.5        # rad: sample's tool axis must be this close to the needed one


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
    # gripper angles from the probe's sweep (jaw<->finger distance 0.059 at -0.17, 0.035 at 1.75)
    gripper_open_q: float = -0.1
    gripper_close_q: float = 1.75
    arm_home: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, -0.1)
    home_noise: float = 0.05
    lift_home_range: tuple = (0.05, 0.30)
    # --- grasp geometry (shared by env reward/reachability and the expert; points come from OBJECT_SPECS)
    # tool axis pitch down from horizontal per grasp style (the heading follows the arm, except for a handle)
    rim_pitch: float = 0.9
    stem_pitch: float = 1.0
    handle_pitch: float = 0.7
    # --- bimanual: these objects are ALWAYS carried with both arms, every other object with one ("rim" style only)
    bimanual_objects: tuple = ("plate", "bowl")
    bimanual_y_max: float = 0.10       # |y| limit of the object so both arms can reach it
    bimanual_rim_angle: float = 0.8    # rad: where each hand grips the rim, measured from the robot-facing point
                                       # (pi/2 = left/right sides; smaller = closer to the robot, more reach)
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
