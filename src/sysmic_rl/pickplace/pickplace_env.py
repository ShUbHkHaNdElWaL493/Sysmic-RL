"""Pick-and-place Gymnasium environment for the Sysmic bimanual robot.

* Objects START on a flat surface of the robot itself (auto-detected "extrusion", see surface.py).
* They must be PLACED on a destination table that is spawned in the arms' reach every episode.
* Per episode either ONE arm or BOTH arms are active (plates/bowls are carried with both by default).

Action layout (13, same as SysmicV2Env): [lift, left(5 joints + gripper), right(5 + gripper)].
"""
from __future__ import annotations

import warnings

import gymnasium as gym
import mujoco
import numpy as np

from .config import ARMS, PickPlaceConfig
from .objects import make_object_specs
from .scene_builder import build_scene_xml
from .surface import detect_support_surface

SIDES = ("left", "right")
DOWN = np.array([0.0, 0.0, -1.0])


class SysmicPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: PickPlaceConfig | None = None, render_mode=None):
        super().__init__()
        self.cfg = cfg or PickPlaceConfig()
        sc, tc = self.cfg.scene, self.cfg.task
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_path(str(build_scene_xml(sc)))
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10.0
        self.data = mujoco.MjData(self.model)
        self.arms = ARMS
        self.specs = make_object_specs(sc.object_scale)
        self.object_names = tuple(sc.objects)
        self._init_indices()

        # episode state (defaults so the first calls work)
        self.step_count = 0
        self.sides = ("left",)
        self.side = "left"
        self.obj_name = self.object_names[0]
        self.table_center = np.array([sum(sc.table_x) / 2, sum(sc.table_y) / 2])
        self.place_z = sum(sc.table_z) / 2
        self.src_z = self.place_z
        self.fail_z = 0.1
        self.goal_pos = np.array([*self.table_center, self.place_z])
        self.target = np.zeros(13)
        self.smoothed = np.zeros(13, dtype=np.float32)
        self.ever_grasped, self.hold, self.rest_com_z = False, 0, 0.0
        self.reach_fallback = False
        self._baseline_pairs = set()
        self._set_active(self.obj_name, ("left",))
        self._set_home()
        mujoco.mj_forward(self.model, self.data)

        self._init_source()
        self._cloud = self._build_reach_cloud() if sc.reach_check else None
        self._compute_baseline_contacts()

        self.action_space = gym.spaces.Box(-1.0, 1.0, (13,), np.float32)
        parts = self._obs_parts()
        self.obs_layout, i = {}, 0
        for k, v in parts:
            self.obs_layout[k] = slice(i, i + len(v))
            i += len(v)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (i,), np.float32)

    def _init_indices(self):
        m, tc = self.model, self.cfg.task
        names = ["lift_joint"] + list(ARMS["left"].joints) + list(ARMS["right"].joints)
        self.q_idx = np.array([m.joint(n).qposadr[0] for n in names])
        self.dof_idx = np.array([m.joint(n).dofadr[0] for n in names])
        j2a = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.act_ids = np.array([j2a[m.joint(n).id] for n in names])
        self.lo = np.array([m.jnt_range[m.joint(n).id, 0] for n in names], dtype=np.float64)
        self.hi = np.array([m.jnt_range[m.joint(n).id, 1] for n in names], dtype=np.float64)
        a = self.act_ids[0]
        if m.actuator_ctrllimited[a]:
            c0, c1 = m.actuator_ctrlrange[a]
            self.lo[0], self.hi[0] = max(self.lo[0], c0), min(self.hi[0], c1)
        self.sl = {"left": slice(1, 7), "right": slice(7, 13)}
        scale = np.full(13, tc.arm_action_scale)
        lag = np.full(13, tc.arm_max_lag)
        scale[0], lag[0] = tc.lift_action_scale, tc.lift_max_lag
        for s in SIDES:
            scale[self.sl[s].stop - 1] = tc.gripper_action_scale
            lag[self.sl[s].stop - 1] = tc.gripper_max_lag
        self.act_scale, self.act_lag = scale, lag
        self.lift_dof = int(self.dof_idx[0])

        self.finger_bid = {s: m.body(ARMS[s].finger_body).id for s in SIDES}
        self.jaw_bid = {s: m.body(ARMS[s].jaw_body).id for s in SIDES}
        gb = m.geom_bodyid
        self.geom_arm = np.zeros(m.ngeom, dtype=np.int8)
        for code, s in enumerate(SIDES, 1):
            for b in range(m.nbody):
                if m.body(b).name.startswith(f"arm_{s}_"):
                    self.geom_arm[gb == b] = code
        self.geom_finger = {s: gb == self.finger_bid[s] for s in SIDES}
        self.geom_jaw = {s: gb == self.jaw_bid[s] for s in SIDES}
        self.geom_tip = np.zeros(m.ngeom, dtype=bool)
        for s in SIDES:
            self.geom_tip |= self.geom_finger[s] | self.geom_jaw[s]

        self.table_bid = m.body("table").id
        self.table_geom = m.geom("table_top").id
        self.ped_geom = m.geom("table_pedestal").id

        self.geom_obj = np.full(m.ngeom, -1, dtype=np.int16)
        self.obj_body, self.obj_qadr, self.obj_dadr = {}, {}, {}
        for i, n in enumerate(self.object_names):
            b = m.body(f"obj_{n}").id
            j = m.joint(f"obj_{n}_joint")
            self.obj_body[n], self.obj_qadr[n], self.obj_dadr[n] = b, int(j.qposadr[0]), int(j.dofadr[0])
            self.geom_obj[gb == b] = i
        self._fric0 = m.geom_friction.copy()
        self._mass0 = m.body_mass.copy()
        self._inertia0 = m.body_inertia.copy()
        try:
            self._goal_mocap = int(m.body("left_target").mocapid[0])
        except Exception:
            self._goal_mocap = -1

    def _init_source(self):
        hide = (self.geom_arm > 0) | (self.geom_obj >= 0)
        hide[self.table_geom] = hide[self.ped_geom] = True
        self.surface = detect_support_surface(self.model, self.data, hide, self.cfg.source)
        gb = self.model.geom_bodyid
        self.geom_surface = (gb == self.surface.body_id) & (self.geom_arm == 0) & (self.geom_obj < 0)
        self.geom_surface[self.table_geom] = False
        # Does the surface ride on the lift? (e.g. part of the lift carriage). If so reach is judged in the carriage frame.
        d, m = self.data, self.model
        saved = d.qpos[self.q_idx[0]]
        zs = []
        for q in (self.lo[0], self.hi[0]):
            d.qpos[self.q_idx[0]] = q
            mujoco.mj_kinematics(m, d)
            zs.append(d.xpos[self.surface.body_id][2])
        d.qpos[self.q_idx[0]] = saved
        mujoco.mj_forward(m, d)
        self.surface_moves = abs(zs[1] - zs[0]) > 1e-3

    def _set_active(self, obj_name, sides):
        self.obj_name, self.sides, self.side = obj_name, tuple(sides), sides[0]
        self.active_idx = self.object_names.index(obj_name)
        self.spec = self.specs[obj_name]
        self.action_mask = np.zeros(13, dtype=np.float32)
        self.action_mask[0] = 1.0
        for s in self.sides:
            self.action_mask[self.sl[s]] = 1.0

    def _set_home(self, rng=None):
        tc = self.cfg.task
        q = np.zeros(13)
        home = np.array(tc.arm_home, dtype=np.float64)
        for s in SIDES:
            h = home.copy()
            if rng is not None:
                h[:5] += rng.uniform(-tc.home_noise, tc.home_noise, 5)
            q[self.sl[s]] = h
        q[0] = rng.uniform(*tc.lift_home_range) if rng is not None else tc.lift_home_range[0]
        q = np.clip(q, self.lo, self.hi)
        self.data.qpos[self.q_idx] = q
        self.target = q.copy()
        self.data.ctrl[self.act_ids] = q

    # ---------------------------------------------------------- kinematics API
    def grasp_point(self, side=None):
        side = side or self.side
        b = self.finger_bid[side]
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ np.array(ARMS[side].grasp_offset)

    def tool_axis_world(self, side=None):
        side = side or self.side
        b = self.finger_bid[side]
        return self.data.xmat[b].reshape(3, 3) @ np.array(ARMS[side].tool_axis)

    def arm_dofs(self, side): return self.dof_idx[self.sl[side]]
    def joint_pos(self): return self.data.qpos[self.q_idx].copy()
    def object_pos(self): return self.data.xpos[self.obj_body[self.obj_name]].copy()
    def object_com(self): return self.data.xipos[self.obj_body[self.obj_name]].copy()

    def object_tilt(self):
        R = self.data.xmat[self.obj_body[self.obj_name]].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1, 1)))

    def object_speed(self):
        return float(np.linalg.norm(self.data.cvel[self.obj_body[self.obj_name]][3:]))

    # ------------------------------------------------------------ grasp plan
    def grasp_plan(self, obj_pos=None, sides=None):
        """{side: (grasp_point_world, tool_axis_world)} for an object whose base-centre is at obj_pos.

        Single arm : 'top' objects -> rim on the robot-facing side, tool down; 'side' objects -> body, tool forward.
        Both arms  : left hand takes the +y side, right hand the -y side (see TaskConfig.bimanual_style).
        Used by the reward (reach term), the observation, the reachability filter and the expert.
        """
        sides = tuple(sides or self.sides)
        o = self.object_pos() if obj_pos is None else np.asarray(obj_pos, dtype=np.float64)
        spec, tc = self.spec, self.cfg.task
        top = spec.grasp_style == "top"
        z = o[2] + (spec.height - tc.top_depth if top else tc.side_height_frac * spec.height)
        if len(sides) == 2:
            r = spec.rim_radius if top else spec.radius
            plan = {}
            for s, sgn in (("left", 1.0), ("right", -1.0)):
                axis = DOWN if tc.bimanual_style == "down" else np.array([0.0, -sgn, 0.0])
                plan[s] = (np.array([o[0], o[1] + sgn * r, z]), axis)
            return plan
        s = sides[0]
        if top:
            return {s: (np.array([o[0] - spec.rim_radius, o[1], z]), DOWN)}
        return {s: (np.array([o[0], o[1], z]), np.array([1.0, 0.0, 0.0]))}

    # --------------------------------------------------------------- reachability
    def _build_reach_cloud(self):
        """Forward-kinematics samples (grasp point + tool axis) over random lift/arm joints. Cheap: kinematics only."""
        m, d, sc = self.model, self.data, self.cfg.scene
        rng = np.random.default_rng(0)
        saved = d.qpos.copy()
        n = sc.reach_samples
        cloud = {}
        for s in SIDES:
            idx = self.q_idx[self.sl[s]][:5]
            lo, hi = self.lo[self.sl[s]][:5], self.hi[self.sl[s]][:5]
            lifts = rng.uniform(self.lo[0], self.hi[0], n)
            qs = rng.uniform(lo, hi, (n, 5))
            pts, axes = np.empty((n, 3)), np.empty((n, 3))
            for i in range(n):
                d.qpos[self.q_idx[0]] = lifts[i]
                d.qpos[idx] = qs[i]
                mujoco.mj_kinematics(m, d)
                pts[i], axes[i] = self.grasp_point(s), self.tool_axis_world(s)
            cloud[s] = (pts, axes, lifts, qs)
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return cloud

    def _carriage_shift(self, lifts, source):
        """z-shift turning 'sample at lift L' into 'same joints at the current lift' (only for carriage-mounted sources)."""
        if source and self.surface_moves:
            return np.stack([np.zeros_like(lifts), np.zeros_like(lifts), lifts - self.joint_pos()[0]], axis=1)
        return 0.0

    def reachable(self, plan, sides=None, source=False):
        """Can the active arm(s) reach this grasp plan?  source=True: the object sits on the (possibly lift-mounted) source surface."""
        if self._cloud is None:
            return True
        sc = self.cfg.scene
        for s in (sides or self.sides):
            pt, axis = plan[s]
            pts, axes, lifts, _ = self._cloud[s]
            pts = pts - self._carriage_shift(lifts, source)
            sel = axes @ axis > np.cos(sc.reach_axis_tol)
            if not sel.any() or np.min(np.linalg.norm(pts[sel] - pt, axis=1)) > sc.reach_tol:
                return False
        return True

    def seed_config(self, goals, source=True):
        """Global-IK seed: best sampled joint configuration for {side: (point, axis)}.

        Returns (lift, {side: q5}, err) or None. Carriage-mounted source (surface_moves): the lift is kept at its
        current value and samples are compared in the carriage frame. Otherwise the second arm's sample is chosen
        with a similar lift (the lift joint is shared).
        """
        if self._cloud is None:
            return None
        ang = lambda axes, ax: np.arccos(np.clip(axes @ ax, -1, 1))
        sides = list(goals)
        carriage = source and self.surface_moves
        now = float(self.joint_pos()[0])

        def err(side):
            pts, axes, lifts, qs = self._cloud[side]
            pts = pts - self._carriage_shift(lifts, source)
            return np.linalg.norm(pts - goals[side][0], axis=1) + 0.10 * ang(axes, goals[side][1])

        e1 = err(sides[0])
        qs1, lifts1 = self._cloud[sides[0]][3], self._cloud[sides[0]][2]
        if len(sides) == 1:
            i = int(np.argmin(e1))
            return (now if carriage else float(lifts1[i])), {sides[0]: qs1[i]}, float(e1[i])
        e2 = err(sides[1])
        lifts2, qs2 = self._cloud[sides[1]][2], self._cloud[sides[1]][3]
        if carriage:
            i, j = int(np.argmin(e1)), int(np.argmin(e2))
            return now, {sides[0]: qs1[i], sides[1]: qs2[j]}, float(e1[i] + e2[j])
        best = None
        for i in np.argsort(e1)[:100]:
            tot = e2 + 5.0 * np.abs(lifts2 - lifts1[i])
            j = int(np.argmin(tot))
            if best is None or e1[i] + tot[j] < best[0]:
                best = (float(e1[i] + tot[j]), int(i), j)
        _, i, j = best
        return float(lifts1[i]), {sides[0]: qs1[i], sides[1]: qs2[j]}, best[0]

    # ---------------------------------------------------------------- contacts
    def _grasp_flags(self, side):
        d = self.data
        n = d.ncon
        if n == 0:
            return False, False
        g1, g2 = d.contact.geom[:n, 0], d.contact.geom[:n, 1]
        touching = d.contact.dist[:n] < 0.001
        o1, o2 = self.geom_obj[g1] == self.active_idx, self.geom_obj[g2] == self.active_idx
        f, j = self.geom_finger[side], self.geom_jaw[side]
        fin = np.any(touching & ((o1 & f[g2]) | (o2 & f[g1])))
        jaw = np.any(touching & ((o1 & j[g2]) | (o2 & j[g1])))
        return bool(fin), bool(jaw)

    def is_grasping(self):
        """True when EVERY active arm has finger+jaw contact with the object."""
        return all(all(self._grasp_flags(s)) for s in self.sides)

    def _bad_contacts(self, ignore_baseline=True):
        """Arm contacts that are neither with the active object nor expected at rest."""
        d, gb = self.data, self.model.geom_bodyid
        n = d.ncon
        out = []
        if n == 0:
            return out
        g, dist = d.contact.geom[:n], d.contact.dist[:n]
        for i in np.where((self.geom_arm[g[:, 0]] > 0) | (self.geom_arm[g[:, 1]] > 0))[0]:
            g1, g2 = int(g[i, 0]), int(g[i, 1])
            if self.geom_obj[g1] >= 0 or self.geom_obj[g2] >= 0:
                continue
            # fingertips may touch the table / the source surface (needed to pinch a plate lying flat)
            for a, b in ((g1, g2), (g2, g1)):
                if self.geom_tip[a] and (b == self.table_geom or self.geom_surface[b]):
                    break
            else:
                if ignore_baseline and frozenset((int(gb[g1]), int(gb[g2]))) in self._baseline_pairs:
                    continue
                out.append(float(dist[i]))
        return out

    def _compute_baseline_contacts(self):
        """Contacts present at the home pose are mesh-overlap artefacts: don't penalise them."""
        self.reset(seed=0, options={"_baseline": True})
        gb, d = self.model.geom_bodyid, self.data
        pairs = set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i, 0]), int(d.contact.geom[i, 1])
            if (self.geom_arm[g1] > 0 or self.geom_arm[g2] > 0) and self.geom_obj[g1] < 0 and self.geom_obj[g2] < 0:
                pairs.add(frozenset((int(gb[g1]), int(gb[g2]))))
        self._baseline_pairs = pairs

    # ------------------------------------------------------------ observation
    def _obs_parts(self):
        d = self.data
        qn = 2 * (self.joint_pos() - self.lo) / (self.hi - self.lo) - 1
        qv = np.clip(d.qvel[self.dof_idx] / 5.0, -5, 5)
        gp = {s: self.grasp_point(s) for s in SIDES}
        quat = {s: d.xquat[self.finger_bid[s]] for s in SIDES}
        b = self.obj_body[self.obj_name]
        com = d.xipos[b]
        onehot = np.zeros(len(self.object_names))
        onehot[self.active_idx] = 1
        plan = self.grasp_plan()
        plan_rel = np.concatenate([plan[s][0] - gp[s] if s in plan else np.zeros(3) for s in SIDES])
        flags = np.array([f for s in SIDES for f in self._grasp_flags(s)], dtype=np.float64)
        return [
            ("q_norm", qn), ("qvel", qv),
            ("grasp_pos", np.r_[gp["left"], gp["right"]]),
            ("grasp_quat", np.r_[quat["left"], quat["right"]]),
            ("prev_action", self.smoothed),
            ("obj_pos", d.xpos[b]), ("obj_quat", d.xquat[b]),
            ("obj_linvel", d.cvel[b][3:]), ("obj_angvel", d.cvel[b][:3]),
            ("obj_type", onehot),
            ("obj_rel_ee", np.r_[com - gp["left"], com - gp["right"]]),
            ("plan_rel", plan_rel),
            ("goal_pos", self.goal_pos), ("goal_rel_obj", self.goal_pos - d.xpos[b]),
            ("active_arm", np.array([s in self.sides for s in SIDES], dtype=np.float64)),
            ("contacts", flags),   # [left finger, left jaw, right finger, right jaw]
        ]

    def _get_obs(self):
        return np.concatenate([v for _, v in self._obs_parts()]).astype(np.float32)

    # ------------------------------------------------------------------ table
    def _place_table(self, centre, top):
        sc = self.cfg.scene
        m = self.model
        m.body_pos[self.table_bid] = [centre[0], centre[1], top]
        h = (top - sc.table_thickness) / 2
        m.geom_size[self.ped_geom, 2] = max(h, 0.01)
        m.geom_pos[self.ped_geom, 2] = -sc.table_thickness - h
        self.table_center, self.place_z = np.array(centre, dtype=np.float64), float(top)

    def _table_collides(self, clearance=None):
        """True if the table (inflated by `clearance`, default SceneConfig.table_clearance) touches any robot geom."""
        m, d = self.model, self.data
        m.geom_margin[self.table_geom] = self.cfg.scene.table_clearance if clearance is None else clearance
        mujoco.mj_forward(m, d)
        hit = False
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i, 0]), int(d.contact.geom[i, 1])
            if self.table_geom in (g1, g2):
                other = g2 if g1 == self.table_geom else g1
                if self.geom_obj[other] < 0 and other != self.ped_geom:
                    hit = True
                    break
        m.geom_margin[self.table_geom] = 0.0
        return hit

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opt = options or {}
        m, d, sc, tc, rng = self.model, self.data, self.cfg.scene, self.cfg.task, self.np_random
        mujoco.mj_resetData(m, d)

        name = opt.get("object") or str(rng.choice(self.object_names))
        forced = opt.get("arm")
        bimanual = forced == "both" or (forced is None and name in tc.bimanual_objects
                                        and rng.random() < tc.bimanual_prob)
        self._set_active(name, ("left", "right") if bimanual else ("left",))
        self._set_home(None if opt.get("_baseline") else rng)
        mujoco.mj_forward(m, d)
        self.reach_fallback = False
        if opt.get("_baseline"):
            return self._get_obs(), {}
        spec = self.spec

        # 1) object start pose on the robot's own surface (reachable by the arm(s) that will be used)
        fixed_xy = opt.get("object_pos")
        pos, sides = None, None
        for attempt in range(300):
            p = (np.array([fixed_xy[0], fixed_xy[1], self.surface.bbox(d)[2]]) if fixed_xy is not None
                 else self.surface.sample(rng, spec.radius, d))
            if bimanual and fixed_xy is None and abs(p[1]) > tc.bimanual_y_max:
                continue
            if bimanual:
                cand = ("left", "right")
            elif forced in SIDES:
                cand = (forced,)
            elif tc.arm_mode == "auto":
                cand = ("left",) if p[1] >= 0 else ("right",)
            else:
                cand = (tc.arm_mode,)
            pos, sides = p, cand
            if fixed_xy is not None or self.reachable(self.grasp_plan(p, cand), cand, source=True):
                break
        else:
            self.reach_fallback = True
        if pos is None:   # every candidate was rejected by the bimanual y-limit: take any spot
            pos = self.surface.sample(rng, spec.radius, d)
            sides = ("left", "right") if bimanual else ("left" if pos[1] >= 0 else "right",)
            sides = tuple(sides)
        self._set_active(name, sides)

        # 2) destination table + goal, in reach of the same arm(s), not touching the robot
        goal_ok = False
        for attempt in range(300):
            c = np.array([rng.uniform(*sc.table_x), rng.uniform(*sc.table_y)])
            top = float(rng.uniform(*sc.table_z))
            if "table_pos" in opt:
                c, top = np.array(opt["table_pos"][:2], dtype=float), float(opt["table_pos"][2])
            self._place_table(c, top)
            if "table_pos" not in opt and self._table_collides():
                continue
            half = np.maximum(np.array(sc.table_half_xy) - spec.radius - 0.03, 0.0)
            gxy = np.array(opt["goal_pos"][:2]) if "goal_pos" in opt else c + rng.uniform(-1, 1, 2) * half
            if np.linalg.norm(gxy - pos[:2]) < tc.min_goal_dist and abs(top - pos[2]) < 0.02:
                continue
            goal = np.array([gxy[0], gxy[1], top])
            if "goal_pos" in opt or "table_pos" in opt or self.reachable(self.grasp_plan(goal, sides), sides):
                goal_ok = True
                break
        if not goal_ok:
            self.reach_fallback = True
            warnings.warn("No reachable table pose found; using the last candidate. Widen SceneConfig.table_* "
                          "or run the probe.", RuntimeWarning, stacklevel=2)
            goal = np.array([*self.table_center, self.place_z])
        self.goal_pos = goal
        self.src_z = float(pos[2])
        self.fail_z = min(self.place_z, self.src_z) - 0.25

        # 3) physics randomisation + place the object
        m.geom_friction[:] = self._fric0
        m.body_mass[:] = self._mass0
        m.body_inertia[:] = self._inertia0
        if tc.randomize_physics:
            b = self.obj_body[name]
            gsel = self.geom_obj == self.active_idx
            m.geom_friction[gsel, 0] = self._fric0[gsel, 0] * rng.uniform(*tc.friction_scale)
            s = rng.uniform(*tc.mass_scale)
            m.body_mass[b] = self._mass0[b] * s
            m.body_inertia[b] = self._inertia0[b] * s
        yaw = float(opt.get("object_yaw", rng.uniform(-0.3, 0.3) if spec.grasp_style == "side"
                            else rng.uniform(-np.pi, np.pi)))
        qa = self.obj_qadr[name]
        d.qpos[qa:qa + 7] = [pos[0], pos[1], pos[2] + 0.002, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        mujoco.mj_forward(m, d)
        for _ in range(30):
            mujoco.mj_step(m, d)
        if self._goal_mocap >= 0:
            d.mocap_pos[self._goal_mocap] = self.goal_pos + np.array([0, 0, 0.03])
        self.step_count, self.hold, self.ever_grasped = 0, 0, False
        self.smoothed = np.zeros(13, dtype=np.float32)
        self.rest_com_z = float(self.object_com()[2])
        mujoco.mj_forward(m, d)
        arm = "both" if len(sides) == 2 else sides[0]
        return self._get_obs(), {"object": name, "arm": arm, "reach_fallback": self.reach_fallback}

    # ------------------------------------------------------------------- step
    def step(self, action):
        tc = self.cfg.task
        m, d = self.model, self.data
        self.step_count += 1
        raw = np.clip(np.asarray(action, dtype=np.float32), -1, 1) * self.action_mask

        prev_sm = self.smoothed.copy()
        self.smoothed = tc.ema_alpha * raw + (1 - tc.ema_alpha) * prev_sm

        prev_t = self.target.copy()
        cur = self.joint_pos()
        nxt = prev_t + self.smoothed * self.act_scale
        nxt = np.clip(nxt, cur - self.act_lag, cur + self.act_lag)
        self.target = np.clip(nxt, self.lo, self.hi)

        for sub in range(tc.frame_skip):
            a = (sub + 1) / tc.frame_skip
            d.ctrl[self.act_ids] = (1 - a) * prev_t + a * self.target
            mujoco.mj_step(m, d)

        obs = self._get_obs()
        info = self._reward_and_flags(raw, prev_sm)
        reward = info.pop("_reward")
        terminated = bool(info["success"] or info["failed"])
        truncated = bool(self.step_count >= tc.max_episode_steps) and not terminated
        info["TimeLimit.truncated"] = truncated
        info["is_success"] = info["success"]
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def _reward_and_flags(self, raw, prev_sm):
        tc, rw = self.cfg.task, self.cfg.reward
        d = self.data
        com, pos = self.object_com(), self.object_pos()
        plan = self.grasp_plan()
        d_reach = float(np.mean([np.linalg.norm(self.grasp_point(s) - plan[s][0]) for s in self.sides]))
        flags = {s: self._grasp_flags(s) for s in self.sides}
        grasped = all(f and j for f, j in flags.values())
        touching = any(f or j for f, j in flags.values())
        self.ever_grasped |= grasped
        lift_h = com[2] - self.rest_com_z
        e_xy = float(np.linalg.norm(pos[:2] - self.goal_pos[:2]))
        e_z = float(abs(pos[2] - self.goal_pos[2]))
        e_goal = float(np.linalg.norm(pos - self.goal_pos))
        tilt, speed = self.object_tilt(), self.object_speed()
        released = not touching
        placed = (e_xy < tc.place_tol_xy and e_z < tc.place_tol_z
                  and tilt < tc.place_max_tilt and speed < tc.place_max_speed)
        self.hold = self.hold + 1 if (placed and released) else 0
        success = self.hold >= tc.hold_steps
        failed = bool(pos[2] < self.fail_z or tilt > tc.fail_tilt)

        bad = self._bad_contacts()
        pen = sum(max(0.0, -x) for x in bad)
        r_col = -min(rw.collision_cap, rw.collision * len(bad) + rw.collision_pen * pen)
        qv = d.qvel[self.dof_idx]
        terms = {
            "reach": rw.reach * (1 - np.tanh(rw.reach_scale * d_reach)),
            "grasp": rw.grasp * float(grasped),
            "lift": rw.lift * float(np.clip(lift_h / tc.lift_target, 0, 1)) * float(grasped),
            "goal": rw.goal * (1 - np.tanh(rw.goal_scale * e_goal)) * float(self.ever_grasped),
            "place": rw.place * float(placed),
            "release": rw.release * float(placed and released),
            "success": rw.success * float(success),
            "collision": r_col,
            "action_rate": -rw.action_rate * float(np.sum((raw - prev_sm) ** 2)),
            "velocity": -rw.velocity * float(np.sum(qv ** 2)),
            "fail": -rw.fail * float(failed),
        }
        return {
            "_reward": sum(terms.values()), "reward_terms": terms,
            "success": bool(success), "failed": failed, "grasped": bool(grasped), "lifted": bool(lift_h > 0.015),
            "placed": bool(placed), "released": bool(released), "err_reach": d_reach, "err_goal_xy": e_xy,
            "tilt": tilt, "n_bad_contacts": len(bad), "object": self.obj_name,
            "arm": "both" if len(self.sides) == 2 else self.side,
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [0.3, 0, 0.8]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.4, 145, -25
            self._renderer.update_scene(self.data, camera=self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            import mujoco.viewer
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
