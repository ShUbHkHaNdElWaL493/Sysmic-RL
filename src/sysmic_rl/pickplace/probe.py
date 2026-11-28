"""Calibration probe: run this BEFORE training and fix SceneConfig/TaskConfig from its output.

  uv run python -m sysmic_rl.pickplace.probe [--scene-xml mjcf/sysmic_v2_scene.xml] [--samples 20000]

Reports (0) the shelf the utensils start on and how much of it / of the table poses is reachable per object,
(1) the gripper calibration (grasp offset, tool axis, closing axis -> ArmSpec), (2) which gripper angle is open vs
closed, (3) mesh-overlap contacts at the home pose, (4) the reachable grasp-point envelope per arm.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from .config import ARMS, PickPlaceConfig
from .pickplace_env import SIDES, SysmicPickPlaceEnv


def pct(a, q): return np.percentile(a, q, axis=0).round(3)


def calibrate_gripper(scene_xml, side, rod_r=0.005, depth=0.12):
    """Gripper geometry in the finger-body frame, from the collision shapes (meshes collide as convex hulls).

    tool_axis  = wrist-roll joint axis (the fingers extend along it).
    close_axis = direction the jaw moves at the grasp point when the gripper closes (q increasing).
    grasp_offset = where a rod of radius rod_r, `depth` m along the fingers, rests against the fixed finger's hull
    (the jaw pushes it there). jaw_q = gripper angle at which the jaw touches it (must be < gripper_close_q).
    """
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
    jq, rg = m.joint(arm.joints[5]).qposadr[0], m.geom("calib_rod").id
    ra = m.jnt_qposadr[m.body("calib_rod").jntadr[0]]
    d.qpos[jq] = m.jnt_range[m.joint(arm.joints[5]).id][0]
    mujoco.mj_kinematics(m, d)
    R, p0 = d.xmat[fb].reshape(3, 3).copy(), d.xpos[fb].copy()
    tool = m.jnt_axis[m.joint(arm.joints[4]).id].copy()                        # wrist roll axis, finger frame
    jid = m.joint(arm.joints[5]).id
    hinge = R.T @ (d.xanchor[jid] - p0)
    axis = R.T @ d.xaxis[jid]
    pt = depth * tool
    close = np.cross(axis, pt - hinge)
    close = close - (close @ tool) * tool
    close /= np.linalg.norm(close)
    binormal = np.cross(tool, close)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([R @ tool, R @ np.cross(binormal, tool), R @ binormal]).flatten())
    d.qpos[ra + 3:ra + 7] = q
    rest = None
    for s_ in np.arange(-0.02, 0.06, 0.0002):                                  # slide off the finger along -close
        cand = pt - close * s_
        d.qpos[ra:ra + 3] = p0 + R @ cand
        mujoco.mj_kinematics(m, d)
        if mujoco.mj_geomDistance(m, d, rg, fg, 0.05, None) > 0:
            rest = cand
            break
    jaw_q = None
    for gq in np.arange(m.jnt_range[jid][0], m.jnt_range[jid][1], 0.005):
        d.qpos[jq] = gq
        mujoco.mj_kinematics(m, d)
        if mujoco.mj_geomDistance(m, d, rg, jg, 0.05, None) <= 0:
            jaw_q = float(gq)
            break
    return {"grasp_offset": rest, "tool_axis": tool, "close_axis": close, "jaw_q": jaw_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-xml", default=None)
    ap.add_argument("--samples", type=int, default=20000)
    a = ap.parse_args()
    cfg = PickPlaceConfig()
    if a.scene_xml:
        cfg.scene.scene_xml = a.scene_xml
    env = SysmicPickPlaceEnv(cfg)
    env.reset(seed=0)
    m, d, rng = env.model, env.data, np.random.default_rng(0)

    print("== shelf (all utensils start here; site in the scene XML) ==")
    cx, cy, z, hx, hy = env.shelf
    print(f"  top z={z:.3f}  x[{cx - hx:.3f},{cx + hx:.3f}] y[{cy - hy:.3f},{cy + hy:.3f}]")
    for q in (env.lo[0], env.hi[0]):
        d.qpos[env.q_idx[0]] = q
        mujoco.mj_kinematics(m, d)
        print(f"  lift={q:.3f}: shelf is {d.xpos[m.body('arm_left_base_link').id][2] - z:.3f} m below the arm bases")
    print("  (wrong place? edit the `shelf` site and `shelf_top` geom in the scene XML)")

    print("\n== reachability of shelf spawns and table poses (fraction feasible) ==")
    n = 200
    for name in env.object_names:
        both = name in cfg.task.bimanual_objects
        ok = 0
        for _ in range(n):
            p, yaw = env._shelf_row(rng, name, both)[name]
            sd = ("left", "right") if both else (("left",) if p[1] >= 0 else ("right",))
            env._set_active(name, sd)
            ok += env.reachable(env.grasp_plan(p, sd, yaw), sd)
        clear = gok = 0
        for _ in range(n):
            c, tyaw = env._sample_table(rng)
            env._place_table(c, cfg.scene.table_z, tyaw)
            if env._table_collides():
                continue
            clear += 1
            half = np.maximum(env.table_half - env.spec.extent - cfg.scene.goal_margin, 0.0)
            hits = 0
            for _ in range(20):
                g = np.array([*env.table_to_world(rng.uniform(-1, 1, 2) * half), env.place_z])
                hits += env.reachable(env.grasp_plan(g, sd, yaw), sd)
            gok += hits > 0
        print(f"  {name:6s} [{env.spec.grasp_style}, {'both arms' if both else 'one arm'}] shelf {ok / n:.0%} | "
              f"table clear of robot {clear / n:.0%}, + reachable goal on it {gok / n:.0%}")
    print("  Low numbers => change the riser height (scene XML), table_z/table_gap/table_y, rim/stem/handle_pitch / "
          "bimanual_rim_angle, or fix gripper/grasp_offset.")

    print("\n== gripper calibration (finger-body frame; compare with ArmSpec in config.py) ==")
    for s in SIDES:
        c, arm = calibrate_gripper(cfg.scene.scene_xml, s), ARMS[s]
        print(f"  {s}: grasp_offset {np.round(c['grasp_offset'], 4)} (config {arm.grasp_offset})  "
              f"tool_axis {np.round(c['tool_axis'], 3)} (config {arm.tool_axis})")
        print(f"        close_axis {np.round(c['close_axis'], 3)} (config {arm.close_axis})  jaw touches a 1 cm rod at "
              f"q={c['jaw_q']:.2f} (gripper_close_q={cfg.task.gripper_close_q} must be larger to squeeze)")

    print("\n== home-pose baseline contacts (auto-ignored by the reward) ==")
    for p in env._baseline_pairs:
        print("  ", " <-> ".join(m.body(b).name for b in p))

    print("\n== gripper sweep (distance jaw COM <-> finger COM; smaller = closed) ==")
    for s in SIDES:
        sl = env.sl[s]
        res = {}
        for q in (env.lo[sl.stop - 1], env.hi[sl.stop - 1]):
            d.qpos[env.q_idx[sl.stop - 1]] = q
            mujoco.mj_forward(m, d)
            res[round(float(q), 2)] = round(float(np.linalg.norm(d.xipos[env.jaw_bid[s]] - d.xipos[env.finger_bid[s]])), 4)
        print(f"  {s}: {res}   -> set TaskConfig.gripper_open_q / gripper_close_q accordingly")

    print("\n== reachable grasp-point envelope (collision-free random configs) ==")
    for s in SIDES:
        sl = env.sl[s]
        pts = {0.0: [], 1.0: []}
        for lift_frac in pts:
            for _ in range(a.samples // 2):
                q = env.joint_pos()
                q[0] = env.lo[0] + lift_frac * (env.hi[0] - env.lo[0])
                q[sl.start:sl.start + 5] = rng.uniform(env.lo[sl.start:sl.start + 5], env.hi[sl.start:sl.start + 5])
                q[sl.stop - 1] = cfg.task.gripper_open_q
                d.qpos[env.q_idx] = q
                mujoco.mj_forward(m, d)
                if not env._bad_contacts():
                    pts[lift_frac].append(env.grasp_point(s))
        for lf, P in pts.items():
            P = np.array(P)
            if len(P) == 0:
                print(f"  {s} lift={lf}: no collision-free samples"); continue
            print(f"  {s} lift={'max' if lf else '0'}: n={len(P)}  x[5,95]={pct(P[:,0],[5,95])} y[5,95]={pct(P[:,1],[5,95])} z[5,95]={pct(P[:,2],[5,95])}")
        P0 = np.array(pts[0.0])
        if len(P0):
            zlow = np.percentile(P0[:, 2], 5)
            band = P0[P0[:, 2] < zlow + 0.08]
            print(f"  >> {s}: lowest comfortable grasp z ~ {zlow:.3f}; x near that height ~ {pct(band[:,0],[10,90])}, "
                  f"y ~ {pct(band[:,1],[10,90])}")


if __name__ == "__main__":
    main()
