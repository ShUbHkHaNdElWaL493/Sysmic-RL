"""Calibration probe: run this BEFORE training and fix SceneConfig/TaskConfig from its output.

  uv run python -m sysmic_rl.pickplace.probe [--robot-xml mjcf/sysmic_v2.xml] [--samples 20000]

Reports (0) the detected source surface (the robot's own extrusion) and how much of it / of the table range is
reachable per object and arm mode, (1) reachable grasp-point envelope per arm at lift=0 and lift=max, suggested
table height / spawn x-range, (2) which gripper angle is open vs closed, (3) mesh-overlap contacts at the home pose.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from .config import PickPlaceConfig
from .pickplace_env import SIDES, SysmicPickPlaceEnv


def pct(a, q): return np.percentile(a, q, axis=0).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-xml", default=None)
    ap.add_argument("--samples", type=int, default=20000)
    a = ap.parse_args()
    cfg = PickPlaceConfig()
    if a.robot_xml:
        cfg.scene.robot_xml = a.robot_xml
    env = SysmicPickPlaceEnv(cfg)
    env.reset(seed=0)
    m, d, rng = env.model, env.data, np.random.default_rng(0)

    print("== source surface (objects start here) ==")
    lo, hi, z = env.surface.bbox(d)
    print(f"  body={env.surface.body_name}  top z={z:.3f}  x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}]  "
          f"usable area={env.surface.area:.3f} m^2  moves_with_lift={env.surface_moves}")
    print("  (wrong patch? set SourceConfig.manual=(cx, cy, z, hx, hy) or SourceConfig.body=...)")

    print("\n== reachability of object spawn spots and table poses (fraction feasible) ==")
    n = 200
    for name in env.object_names:
        row = []
        for label, sides in (("one-arm", None), ("both-arms", ("left", "right"))):
            ok = tot = 0
            for _ in range(n):
                env._set_active(name, ("left",))
                p = env.surface.sample(rng, env.spec.radius, d)
                sd = sides or (("left",) if p[1] >= 0 else ("right",))
                env._set_active(name, sd)
                tot += 1
                ok += env.reachable(env.grasp_plan(p, sd), sd, source=True)
            row.append(f"{label} {ok / tot:.0%}")
        gok = 0
        for _ in range(n):
            c = np.array([rng.uniform(*cfg.scene.table_x), rng.uniform(*cfg.scene.table_y)])
            top = rng.uniform(*cfg.scene.table_z)
            env._place_table(c, top)
            if env._table_collides():
                continue
            env._set_active(name, ("left", "right") if name in cfg.task.bimanual_objects else ("left",))
            g = np.array([c[0], c[1], top])
            gok += env.reachable(env.grasp_plan(g), env.sides)
        row.append(f"table pose+goal ok {gok / n:.0%}")
        print(f"  {name:6s} [{env.spec.grasp_style}] " + " | ".join(row))
    print("  Low numbers => widen table_* ranges, change the object's grasp_style, or fix gripper/grasp_offset.")

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
            print(f"  >> {s}: lowest comfortable z ~ {zlow:.3f}. Suggest table_top_z ~ {zlow - 0.01:.3f}; "
                  f"x-band near that height ~ {pct(band[:,0],[10,90])} (use for spawn_x), y ~ {pct(band[:,1],[10,90])}")


if __name__ == "__main__":
    main()
