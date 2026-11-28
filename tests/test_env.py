import warnings

import mujoco
import numpy as np
import pytest

from sysmic_rl.imitation.collect import collect
from sysmic_rl.imitation.dataset import load_demos
from sysmic_rl.env import SysmicBotConfig, SysmicBotEnv

warnings.filterwarnings("ignore", message="No reachable table pose")


@pytest.fixture(scope="module")
def cfg():
    c = SysmicBotConfig()          # the real mjcf/sysmic_v2_scene.xml
    c.scene.reach_samples = 20000  # sparser clouds make most resets fall back
    return c


@pytest.fixture(scope="module")
def env(cfg):
    return SysmicBotEnv(cfg)


def test_spaces_and_determinism(cfg):
    e1, e2 = SysmicBotEnv(cfg), SysmicBotEnv(cfg)
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    assert o1.shape == e1.observation_space.shape and np.allclose(o1, o2)
    assert e1.obs_layout["others_pos"].stop - e1.obs_layout["others_pos"].start == 3 * (len(e1.object_names) - 1)
    a = e1.action_space.sample()
    s1, s2 = e1.step(a), e2.step(a)
    assert np.allclose(s1[0], s2[0]) and np.isfinite(s1[1])


@pytest.mark.parametrize("seed", range(5))
def test_all_utensils_settle_on_shelf(env, seed):
    env.reset(seed=seed)
    cx, cy, z, hx, hy = env.shelf
    spans = []
    for n in env.object_names:
        b = env.obj_body[n]
        p = env.data.xpos[b]
        assert abs(p[2] - z) < 0.01, n                      # resting on the shelf (not fallen, not ejected)
        assert abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy, n
        assert env.data.xmat[b][8] > np.cos(0.1), n          # upright
        w = env.specs[n].footprint_y
        spans.append((p[1] - w, p[1] + w))
    spans.sort()
    assert all(a[1] <= b[0] + 1e-3 for a, b in zip(spans, spans[1:]))   # no overlap along the shelf


def _ray(env, obj, origin_local, dir_local):
    """Distance along a ray (object frame) to the first COLLISION geom of that utensil, or None."""
    m, d = env.model, env.data
    b = env.obj_body[obj]
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    geomid = np.zeros(1, dtype=np.int32)
    groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)          # collision geoms are group 3
    dist = mujoco.mj_ray(m, d, p + R @ np.array(origin_local), R @ np.array(dir_local, dtype=float), groups, 1, -1, geomid)
    return None if geomid[0] < 0 or env.geom_obj[geomid[0]] != env.object_names.index(obj) else dist


def test_utensil_shapes(env):
    env.reset(seed=0)
    down = (0, 0, -1)
    # mug: hollow with an 8 mm solid base; the handle loop has a hole and a graspable outer bar
    assert abs(_ray(env, "mug", (0, 0, 0.2), down) - (0.2 - 0.008)) < 1e-3
    assert _ray(env, "mug", (0.0475, -0.05, 0.0475), (0, 1, 0)) is None          # through the handle hole
    assert _ray(env, "mug", (0.060, -0.05, 0.0475), (0, 1, 0)) is not None       # outer bar
    # wine glass: thin stem under a hollow bowl
    assert _ray(env, "glass", (0.008, -0.05, 0.035), (0, 1, 0)) is None          # beside the stem
    assert _ray(env, "glass", (0.0, -0.05, 0.035), (0, 1, 0)) is not None        # the stem
    assert _ray(env, "glass", (0, 0, 0.3), down) > 0.3 - 0.075                    # bowl is hollow down to its bottom
    # bowl / plate: hollow dishes (the centre is low, the rim high)
    assert _ray(env, "bowl", (0, 0, 0.2), down) > 0.2 - 0.006
    assert _ray(env, "plate", (0, 0, 0.2), down) > 0.2 - 0.008


def test_grasp_plans_target_handle_stem_and_rims(env):
    env.reset(seed=1, options={"object": "mug"})
    (p, t, c), = env.grasp_plan().values()
    o, yaw = env.object_pos(), env.object_yaw()
    hd = np.array([np.cos(yaw), np.sin(yaw), 0])
    assert np.allclose(p, o + 0.060 * hd + [0, 0, 0.0475])                       # outer bar of the handle
    assert abs(t @ c) < 1e-9 and abs(c @ hd) < 1e-9                               # jaws close across the loop
    assert env.heading(env.side, p) @ (-hd) > np.cos(0.3)                         # handle aimed at the grasping arm
    env.reset(seed=1, options={"object": "glass"})
    (p, t, c), = env.grasp_plan().values()
    assert np.allclose(p, env.object_pos() + [0, 0, 0.035]) and abs(c[2]) < 1e-9  # stem; jaws close horizontally
    env.reset(seed=1, options={"object": "bowl"})
    o = env.object_pos()
    for s, (p, t, c) in env.grasp_plan().items():
        assert abs(t @ c) < 1e-9 and c @ np.r_[o[:2] - p[:2], 0] > 0              # jaw moves inwards: finger inside


def calibrate_gripper(scene_xml, side, rod_r=0.005, depth=0.12):
    """Gripper geometry in the finger-body frame, from the collision shapes (meshes collide as convex hulls).

    tool_axis  = wrist-roll joint axis (the fingers extend along it).
    close_axis = direction the jaw moves at the grasp point when the gripper closes (q increasing).
    grasp_offset = where a rod of radius rod_r, `depth` m along the fingers, rests against the fixed finger's hull
    (the jaw pushes it there). jaw_q = gripper angle at which the jaw touches it (must be < gripper_close_q).
    """
    from sysmic_rl.env.config import ARMS
    arm = ARMS[side]
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    b = spec.worldbody.add_body(name="calib_rod", pos=[0, 0, 3])
    b.add_freejoint()
    b.add_geom(name="calib_rod", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[rod_r, 0.02, 0])
    m = spec.compile()
    d = mujoco.MjData(m)
    fb, jb = m.body(arm.finger_body).id, m.body(arm.jaw_body).id
    fg = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and m.geom_contype[g])
    jg = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == jb and m.geom_contype[g])
    jid = m.joint(arm.joints[5]).id
    jq, rg = m.jnt_qposadr[jid], m.geom("calib_rod").id
    ra = m.jnt_qposadr[m.body("calib_rod").jntadr[0]]
    d.qpos[jq] = m.jnt_range[jid][0]
    mujoco.mj_kinematics(m, d)
    R, p0 = d.xmat[fb].reshape(3, 3).copy(), d.xpos[fb].copy()
    tool = m.jnt_axis[m.joint(arm.joints[4]).id].copy()
    hinge, axis = R.T @ (d.xanchor[jid] - p0), R.T @ d.xaxis[jid]
    pt = depth * tool
    close = np.cross(axis, pt - hinge)
    close -= (close @ tool) * tool
    close /= np.linalg.norm(close)
    binormal = np.cross(tool, close)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([R @ tool, R @ np.cross(binormal, tool), R @ binormal]).flatten())
    d.qpos[ra + 3:ra + 7] = q
    rest = None
    for s_ in np.arange(-0.02, 0.06, 0.0002):                 # slide the rod off the finger hull along -close
        cand = pt - close * s_
        d.qpos[ra:ra + 3] = p0 + R @ cand
        mujoco.mj_kinematics(m, d)
        if mujoco.mj_geomDistance(m, d, rg, fg, 0.05, None) > 0:
            rest = cand
            break
    jaw_q = None
    for gq in np.arange(*m.jnt_range[jid], 0.005):             # close until the jaw touches it
        d.qpos[jq] = gq
        mujoco.mj_kinematics(m, d)
        if mujoco.mj_geomDistance(m, d, rg, jg, 0.05, None) <= 0:
            jaw_q = float(gq)
            break
    return {"grasp_offset": rest, "tool_axis": tool, "close_axis": close, "jaw_q": jaw_q}


def test_gripper_calibration_matches_config(cfg):
    from sysmic_rl.env.config import ARMS
    for s in ("left", "right"):
        c = calibrate_gripper(cfg.scene.scene_xml, s)
        assert np.allclose(c["grasp_offset"], ARMS[s].grasp_offset, atol=1e-3)
        assert np.allclose(c["tool_axis"], ARMS[s].tool_axis, atol=1e-3)
        assert np.allclose(c["close_axis"], ARMS[s].close_axis, atol=2e-3)
        assert c["jaw_q"] < cfg.task.gripper_close_q          # the jaw reaches the object before its target: it squeezes


def test_utensil_order_is_randomised(env):
    orders = set()
    for seed in range(8):
        env.reset(seed=seed)
        orders.add(tuple(sorted(env.object_names, key=lambda n: env.data.xpos[env.obj_body[n]][1])))
    assert len(orders) > 1


def test_table_in_front_at_fixed_height(env, cfg):
    sc = cfg.scene
    corners = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * env.table_half
    checked = 0
    for seed in range(12):
        _, info = env.reset(seed=seed)
        assert env.place_z == sc.table_z
        assert sc.table_yaw[0] <= env.table_yaw <= sc.table_yaw[1]
        assert env.table_to_world(corners)[:, 0].min() >= env.front_x - 1e-9   # in front of the shelf
        assert not env._table_collides(clearance=0.0)
        if info["reach_fallback"]:
            continue
        checked += 1
        assert abs(env.goal_pos[2] - env.place_z) < 1e-9
        assert np.all(np.abs(env.world_to_table(env.goal_pos[:2])) <= env.table_half)
        assert env.reachable(env.grasp_plan(env.goal_pos), env.sides)
    assert checked >= 6   # fallbacks come from the sparse test cloud; see test_default_workspace_is_reachable


def test_table_options_and_frames(env):
    env.reset(seed=0, options={"table_pos": (0.45, 0.05, 0.8), "table_yaw": 0.7})
    assert np.allclose(env.table_center, (0.45, 0.05)) and env.place_z == 0.8 and env.table_yaw == 0.7
    uv = np.array([0.1, -0.05])
    assert np.allclose(env.world_to_table(env.table_to_world(uv)), uv)
    # the table's top face really is where the env thinks it is
    top = env.data.geom_xpos[env.table_geom][2] + env.model.geom_size[env.table_geom][2]
    assert abs(top - 0.8) < 1e-9
    env.reset(seed=0, options={"table_pos": (0.55, 0.0)})   # (x, y) only -> the configured fixed height
    assert env.place_z == env.cfg.scene.table_z


def test_table_collision_check_sees_the_base(env):
    # a table pushed into the robot's base must be detected (base_link and table are both world-welded,
    # so MuJoCo would not report a contact between them)
    env.reset(seed=0)
    env._place_table((0.15, 0.0), 0.6)
    assert env._table_collides(clearance=0.0)
    env._place_table((1.5, 0.0), 0.6)
    assert not env._table_collides()


def test_bimanual_for_plate_and_bowl_only(env):
    for name, both in [("plate", True), ("bowl", True), ("glass", False), ("mug", False)]:
        for seed in range(3):
            _, info = env.reset(seed=seed, options={"object": name})
            assert (info["arm"] == "both") == both and (len(env.sides) == 2) == both
    with pytest.raises(ValueError):
        env.reset(seed=2, options={"object": "plate", "arm": "left"})
    with pytest.raises(ValueError):
        env.reset(seed=2, options={"object": "glass", "arm": "both"})
    env.reset(seed=2, options={"object": "glass", "arm": "right"})
    assert env.sides == ("right",)


def test_gripper_close_q_is_closed(env):
    d, m, tc = env.data, env.model, env.cfg.task
    gap = {}
    for q in (tc.gripper_open_q, tc.gripper_close_q):
        d.qpos[env.q_idx[env.sl["left"].stop - 1]] = q
        mujoco.mj_kinematics(m, d)
        gap[q] = np.linalg.norm(d.xipos[env.jaw_bid["left"]] - d.xipos[env.finger_bid["left"]])
    assert gap[tc.gripper_close_q] < gap[tc.gripper_open_q]
    env.reset(seed=0)


def test_bimanual_plan_has_opposite_sides(env):
    env.reset(seed=3, options={"object": "plate"})
    plan, o = env.grasp_plan(), env.object_pos()
    assert plan["left"][0][1] > o[1] > plan["right"][0][1]
    assert plan["left"][0][0] < o[0] and plan["right"][0][0] < o[0]      # rim points on the robot-facing half
    assert all(plan[s][1][2] < 0 and plan[s][1][0] > 0 for s in env.sides)   # tool tilted down + forward
    assert set(env.action_mask[1:13]) == {1.0}


def test_touching_inactive_utensil_is_bad_contact(env):
    env.reset(seed=0, options={"object": "plate"})
    d, m = env.data, env.model
    finger = int(np.where(env.geom_finger["left"])[0][0])
    other = int(np.where(env.geom_obj == env.object_names.index("glass"))[0][0])
    active = int(np.where(env.geom_obj == env.active_idx)[0][0])
    for g in (other, active):
        d.ncon = 1
        d.contact.geom[0] = (finger, g)
        d.contact.dist[0] = -0.001
        n_bad = len(env._bad_contacts(ignore_baseline=False))
        assert n_bad == (1 if g == other else 0)
    env.reset(seed=0)   # leave the shared env in a clean state


def test_expert_runs_bimanual_and_single(env):
    from sysmic_rl.imitation.expert import ScriptedExpert
    ex = ScriptedExpert()
    for name in ("plate", "glass"):
        env.reset(seed=1, options={"object": name}); ex.reset()
        for _ in range(60):
            a = ex.act(env)
            assert a.shape == (13,) and np.all(np.abs(a) <= 1.0)
            env.step(a)


def test_inactive_arm_is_frozen(env):
    env.reset(seed=1, options={"arm": "left", "object": "glass"})
    t0 = env.target[env.sl["right"]].copy()
    for _ in range(20):
        env.step(np.ones(13, dtype=np.float32))
    assert np.allclose(env.target[env.sl["right"]], t0)
    assert not np.allclose(env.target[env.sl["left"]], t0)


def test_collect_and_load(cfg, tmp_path):
    out = tmp_path / "d.h5"
    collect(2, str(out), cfg, noise=0.0, seed=0, keep_failures=True, log=lambda *_: None)
    d, n = load_demos(str(out), successful_only=False)
    assert d["obs"].shape[1] == SysmicBotEnv(cfg).observation_space.shape[0]
    assert d["actions"].shape[1] == 13 and len(d["obs"]) == len(d["rewards"])


def test_seed_config(env):
    env.reset(seed=0, options={"object": "plate"})
    plan = env.grasp_plan()
    lift, q5, err = env.seed_config({s: (plan[s][0], plan[s][1]) for s in env.sides})
    assert np.isfinite(err) and env.lo[0] <= lift <= env.hi[0]
    assert set(q5) == set(env.sides) and all(len(q) == 5 for q in q5.values())


def test_default_workspace_is_reachable():
    """With the default config (full reach cloud) every object gets a reachable shelf spot AND table goal. Run this
    after changing the riser, table_z / table_* ranges or the grasp pitches."""
    env = SysmicBotEnv()
    m, d = env.model, env.data
    for name in env.object_names:
        for seed in range(15):
            _, info = env.reset(seed=seed, options={"object": name})
            assert not info["reach_fallback"], (name, seed)
            for i in range(d.ncon):                                   # no utensil spawns inside another / the robot
                g1, g2 = d.contact.geom[i]
                if env.geom_obj[g1] >= 0 or env.geom_obj[g2] >= 0:
                    assert d.contact.dist[i] > -2e-3, (name, seed, m.geom(g1).name, m.geom(g2).name)
            assert np.allclose(d.mocap_pos[env._goal_mocap], env.goal_pos + [0, 0, 0.03])
