"""Pick-and-place Gymnasium environment for the Sysmic bimanual robot.

* The scene is the static `mjcf/sysmic_v2_scene.xml` (robot included unmodified + shelf riser + table + utensils).
* All utensils START in a row on the shelf riser; one of them (the active one) must be moved.
* It must be PLACED on the destination table in front of the robot, re-posed (x, y, yaw; fixed height) every episode.
* Plates/bowls are always carried with BOTH arms (rim pinch), the glass by its stem and the mug by its handle with ONE.
* Every pose is randomised in place (data.qpos / model.body_pos / model.body_quat); the model is compiled once.

Action layout (13, same as SysmicV2Env): [lift, left(5 joints + gripper), right(5 + gripper)].
"""
from __future__ import annotations

import warnings

import gymnasium as gym
import mujoco
import numpy as np

from .config import ARMS, OBJECT_SPECS, SysmicBotConfig

SIDES = ("left", "right")


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _perp(v, axis):
    """Unit component of v perpendicular to the unit vector axis."""
    v = np.asarray(v, dtype=np.float64) - (np.dot(v, axis)) * axis
    return v / np.linalg.norm(v)


class SysmicBotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ------------------------------------------------------------------ init
    def __init__(self, cfg: SysmicBotConfig | None = None, render_mode=None):
        super().__init__()
        self.cfg = cfg or SysmicBotConfig()
        sc, tc = self.cfg.scene, self.cfg.task
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_path(str(sc.scene_xml))
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10.0
        self.data = mujoco.MjData(self.model)
        self.arms = ARMS
        self.specs = OBJECT_SPECS
        self.object_names = tuple(sc.objects)
        self._init_indices()

        # episode state (defaults so the first calls work)
        self.step_count = 0
        self.sides = ("left",)
        self.side = "left"
        self.obj_name = self.object_names[0]
        self.table_center = self.model.body_pos[self.table_bid][:2].copy()
        self.table_yaw = 0.0
        self.place_z = float(sc.table_z)
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
        self.table_half = m.geom_size[self.table_geom][:2].copy()
        table_geoms = gb == self.table_bid

        self.geom_obj = np.full(m.ngeom, -1, dtype=np.int16)
        self.obj_body, self.obj_qadr, self.obj_dadr = {}, {}, {}
        for i, n in enumerate(self.object_names):
            b = m.body(f"obj_{n}").id
            j = m.joint(f"obj_{n}_joint")
            self.obj_body[n], self.obj_qadr[n], self.obj_dadr[n] = b, int(j.qposadr[0]), int(j.dofadr[0])
            self.geom_obj[gb == b] = i
        # robot geoms the table must keep clear of: every collidable geom that is not world/table/utensil
        self.robot_geoms = np.where((gb != 0) & ~table_geoms & (self.geom_obj < 0)
                                    & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))[0]
        self._movable = table_geoms | (self.geom_obj >= 0)   # ignored by the reach cloud's collision filter
        self._fric0 = m.geom_friction.copy()
        self._mass0 = m.body_mass.copy()
        self._inertia0 = m.body_inertia.copy()
        self._goal_mocap = int(m.body("goal").mocapid[0])   # visual goal marker (scene XML)

    def _init_source(self):
        """The shelf (site in the scene XML) on the static base: world-frame rectangle (cx, cy, top_z, hx, hy)."""
        m = self.model
        sid = m.site(self.cfg.scene.shelf_site).id
        (cx, cy, z), (hx, hy, _) = self.data.site_xpos[sid], m.site_size[sid]   # site centre = shelf surface
        self.shelf = (float(cx), float(cy), float(z), float(hx), float(hy))
        # the table must stay in front of the shelf (otherwise it would cover the utensils)
        self.front_x = float(cx + hx) + self.cfg.scene.table_clearance
        # fingertips may touch the shelf / base (needed to pinch a plate lying flat) without penalty
        gb = m.geom_bodyid
        self.geom_surface = (gb == m.body("base_link").id) | (np.arange(m.ngeom) == m.geom("shelf_top").id)
        # where each arm's pan axis meets the ground plan (xy): sets the arm's natural approach heading
        self.pan_xy = {s: self.data.xanchor[m.joint(ARMS[s].joints[0]).id][:2].copy() for s in SIDES}

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

    def close_axis_world(self, side=None):
        side = side or self.side
        b = self.finger_bid[side]
        return self.data.xmat[b].reshape(3, 3) @ np.array(ARMS[side].close_axis)

    def arm_dofs(self, side): return self.dof_idx[self.sl[side]]
    def joint_pos(self): return self.data.qpos[self.q_idx].copy()
    def object_pos(self): return self.data.xpos[self.obj_body[self.obj_name]].copy()
    def object_com(self): return self.data.xipos[self.obj_body[self.obj_name]].copy()

    def object_yaw(self):
        R = self.data.xmat[self.obj_body[self.obj_name]].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def object_tilt(self):
        R = self.data.xmat[self.obj_body[self.obj_name]].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1, 1)))

    def object_speed(self):
        return float(np.linalg.norm(self.data.cvel[self.obj_body[self.obj_name]][3:]))

    # ------------------------------------------------------------ grasp plan
    def heading(self, side, p):
        """Horizontal unit vector from the arm's pan axis to point p: the arm's natural approach heading."""
        v = np.asarray(p[:2], dtype=np.float64) - self.pan_xy[side]
        return np.array([*(v / np.linalg.norm(v)), 0.0])

    def yaw_to_arm(self, p, side):
        """Object yaw that points its local +x (the mug handle) from p towards the arm's pan axis."""
        v = self.pan_xy[side] - np.asarray(p[:2], dtype=np.float64)
        return float(np.arctan2(v[1], v[0]))

    def close_signed(self):
        """Rim grasps need the jaw on the outside of the wall (the thick fixed finger goes inside); others are symmetric."""
        return self.spec.grasp_style == "rim"

    def heading_free(self):
        """Stem/rim grasps only fix the tool pitch: the heading follows the arm. A handle fixes the full direction."""
        return self.spec.grasp_style != "handle"

    def grasp_plan(self, obj_pos=None, sides=None, obj_yaw=None, headings=None):
        """{side: (grasp_point, tool_axis, close_axis)} for the active object with its base centre at obj_pos / yaw obj_yaw
        (default: where it is now). close_axis = direction the jaw must move when closing (perpendicular to tool_axis).

        The arms have 5 joints, so the tool heading is not free: for "rim"/"stem" only the pitch is prescribed and the
        heading is the arm's (default: `heading()`, or `headings[side]`, e.g. the current tool heading).
        rim    : pitch `rim_pitch`; jaw closes across the rim wall from the outside. One arm: the robot-facing point of
                 the rim. Both arms: points `bimanual_rim_angle` either side of it (left hand +y, right hand -y).
        stem   : one arm, pitch `stem_pitch`; jaws close horizontally across the stem.
        handle : one arm; tool along the handle towards the body, pitch `handle_pitch`; the jaws close on both flat sides
                 of the handle loop (outer bar between them).
        Used by the reward (reach term), the observation, the reachability filter and the expert.
        """
        sides = tuple(sides or self.sides)
        o = self.object_pos() if obj_pos is None else np.asarray(obj_pos, dtype=np.float64)
        yaw = self.object_yaw() if obj_yaw is None else float(obj_yaw)
        spec, tc = self.spec, self.cfg.task
        up = np.array([0.0, 0.0, 1.0])
        z = o[2] + spec.grasp_z

        def tool(h, pitch):
            return np.cos(pitch) * h - np.sin(pitch) * up

        def head(s, p):
            return headings[s] if headings and s in headings else self.heading(s, p)

        if spec.grasp_style == "rim":
            angles = {"left": tc.bimanual_rim_angle, "right": -tc.bimanual_rim_angle} if len(sides) == 2 else {sides[0]: 0.0}
            plan = {}
            for s, a in angles.items():
                radial = np.array([-np.cos(a), np.sin(a), 0.0])    # axis -> grasp point (0 = towards the robot, -x)
                p = o + spec.grasp_r * radial
                p[2] = z
                t = tool(head(s, p), tc.rim_pitch)
                # wall normal pointing into the dish: the jaw moves along it from outside (finger inside)
                inward = -np.sin(spec.wall_slope) * radial + np.cos(spec.wall_slope) * up
                plan[s] = (p, t, _perp(inward, t))
            return plan
        if len(sides) != 1:
            raise ValueError(f"{spec.name} ({spec.grasp_style} grasp) is carried with one arm")
        s = sides[0]
        if spec.grasp_style == "stem":
            p = np.array([o[0], o[1], z])
            h = head(s, p)
            return {s: (p, tool(h, tc.stem_pitch), np.cross(up, h))}
        if spec.grasp_style == "handle":
            hd = np.array([np.cos(yaw), np.sin(yaw), 0.0])        # handle direction (object's local +x)
            p = o + spec.grasp_r * hd
            p[2] = z
            return {s: (p, tool(-hd, tc.handle_pitch), np.cross(up, -hd))}
        raise ValueError(f"unknown grasp_style {spec.grasp_style!r}")

    def grasp_axes(self, side, tool_now):
        """(tool_axis, close_axis) targets for an arm whose tool currently points along tool_now (heading kept)."""
        h = np.array([tool_now[0], tool_now[1], 0.0])
        n = np.linalg.norm(h)
        heads = {side: h / n} if self.heading_free() and n > 1e-3 else None
        _, t, c = self.grasp_plan(headings=heads)[side]
        return t, c

    # --------------------------------------------------------------- reachability
    def _build_reach_cloud(self):
        """Forward-kinematics samples (grasp point, tool axis, closing axis) over random lift/arm joints, keeping only
        configurations where the arm touches neither the robot, the other arm (at home) nor the shelf riser.
        Utensils and the table are ignored (they move)."""
        m, d, sc = self.model, self.data, self.cfg.scene
        rng = np.random.default_rng(0)
        saved = d.qpos.copy()
        n = sc.reach_samples
        cloud = {}
        ct, ca = m.geom_contype.copy(), m.geom_conaffinity.copy()
        m.geom_contype[self._movable], m.geom_conaffinity[self._movable] = 0, 0   # ignored anyway: skip them (faster)
        for s in SIDES:
            idx = self.q_idx[self.sl[s]][:5]
            lo, hi = self.lo[self.sl[s]][:5], self.hi[self.sl[s]][:5]
            lifts = rng.uniform(self.lo[0], self.hi[0], n)
            qs = rng.uniform(lo, hi, (n, 5))
            X, M = np.empty((n, 3)), np.empty((n, 3, 3))
            ok = np.ones(n, dtype=bool)
            code = SIDES.index(s) + 1
            b, garm = self.finger_bid[s], self.geom_arm
            d.qpos[:] = saved
            for i in range(n):
                d.qpos[self.q_idx[0]] = lifts[i]
                d.qpos[idx] = qs[i]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                if d.ncon:
                    g = d.contact.geom[:d.ncon]
                    ok[i] = not np.any((d.contact.dist[:d.ncon] < 0) & ((garm[g[:, 0]] == code) | (garm[g[:, 1]] == code)))
                X[i], M[i] = d.xpos[b], d.xmat[b].reshape(3, 3)
            arm = ARMS[s]
            pts = X + M @ np.array(arm.grasp_offset)
            axes, closes = M @ np.array(arm.tool_axis), M @ np.array(arm.close_axis)
            cloud[s] = (pts[ok], axes[ok], lifts[ok], qs[ok], closes[ok])
        m.geom_contype[:], m.geom_conaffinity[:] = ct, ca
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return cloud

    def _axis_err(self, axes, axis):
        """Angle between sampled tool axes and the planned one: full direction for a handle, pitch only otherwise."""
        if self.heading_free():
            return np.abs(np.arcsin(np.clip(axes[:, 2], -1, 1)) - np.arcsin(np.clip(axis[2], -1, 1)))
        return np.arccos(np.clip(axes @ axis, -1, 1))

    def reachable(self, plan, sides=None):
        """Can the active arm(s) reach this grasp plan? Point + tool axis (pitch only when the heading is free); the
        closing direction is free: the wrist roll turns about the tool axis and the grasp point sits ~on that axis."""
        if self._cloud is None:
            return True
        sc = self.cfg.scene
        for s in (sides or self.sides):
            pt, axis = plan[s][0], plan[s][1]
            pts, axes = self._cloud[s][0], self._cloud[s][1]
            sel = self._axis_err(axes, axis) < sc.reach_axis_tol
            if not sel.any() or np.min(np.linalg.norm(pts[sel] - pt, axis=1)) > sc.reach_tol:
                return False
        return True

    def seed_config(self, goals):
        """Global-IK seed: best sampled joint configuration for {side: (point, axis[, close_axis])}.

        Returns (lift, {side: q5}, err) or None. With two arms the second arm's sample is chosen
        with a similar lift (the lift joint is shared). If a close_axis is given, the wrist roll of the
        chosen sample is turned so the jaws close along it.
        """
        if self._cloud is None:
            return None
        sides = list(goals)

        def err(side):
            pts, axes = self._cloud[side][0], self._cloud[side][1]
            return np.linalg.norm(pts - goals[side][0], axis=1) + 0.10 * self._axis_err(axes, goals[side][1])

        e1 = err(sides[0])
        qs1, lifts1 = self._cloud[sides[0]][3], self._cloud[sides[0]][2]
        if len(sides) == 1:
            i = int(np.argmin(e1))
            return float(lifts1[i]), {sides[0]: self._roll_to(sides[0], i, goals[sides[0]])}, float(e1[i])
        e2 = err(sides[1])
        lifts2, qs2 = self._cloud[sides[1]][2], self._cloud[sides[1]][3]
        best = None
        for i in np.argsort(e1)[:100]:
            tot = e2 + 5.0 * np.abs(lifts2 - lifts1[i])
            j = int(np.argmin(tot))
            if best is None or e1[i] + tot[j] < best[0]:
                best = (float(e1[i] + tot[j]), int(i), j)
        _, i, j = best
        return (float(lifts1[i]), {sides[0]: self._roll_to(sides[0], i, goals[sides[0]]),
                                   sides[1]: self._roll_to(sides[1], j, goals[sides[1]])}, best[0])

    def _roll_to(self, side, i, goal):
        """Cloud sample i's 5 arm joints, with the wrist roll turned so the jaws close as the active grasp needs
        (for the sample's own tool heading). The wrist roll joint turns the finger body about its tool axis, so this
        is a pure rotation of close_axis. goal[2], if given, overrides the closing direction."""
        q5 = self._cloud[side][3][i].copy()
        a, c = self._cloud[side][1][i], self._cloud[side][4][i]
        want = goal[2] if len(goal) > 2 and goal[2] is not None else self.grasp_axes(side, a)[1]
        want = _perp(want, a)
        delta = np.arctan2(np.dot(np.cross(c, want), a), np.dot(c, want))
        if not self.close_signed():
            delta = (delta + np.pi / 2) % np.pi - np.pi / 2       # symmetric grasp: turn <= 90 deg
        k = self.sl[side].start + 4
        for dq in (delta, delta + np.pi, delta - np.pi):
            if self.lo[k] <= q5[4] + dq <= self.hi[k]:
                q5[4] += dq
                break
        return q5

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
            if self.geom_obj[g1] == self.active_idx or self.geom_obj[g2] == self.active_idx:
                continue   # touching the ACTIVE utensil is wanted; the other utensils are obstacles
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
        others = np.concatenate([d.xpos[self.obj_body[n]] for n in self.object_names if n != self.obj_name] or [[]])
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
            ("others_pos", others),  # positions of the inactive utensils (obstacles), object_names order
        ]

    def _get_obs(self):
        return np.concatenate([v for _, v in self._obs_parts()]).astype(np.float32)

    # ------------------------------------------------------------------ table
    def _place_table(self, centre, top, yaw=0.0):
        """Re-pose the static table body (top-surface centre + yaw). Legs are fixed-length visuals, nothing is resized."""
        m = self.model
        m.body_pos[self.table_bid] = [centre[0], centre[1], top]
        m.body_quat[self.table_bid] = _yaw_quat(yaw)
        mujoco.mj_kinematics(m, self.data)
        self.table_center, self.place_z, self.table_yaw = np.array(centre, dtype=np.float64), float(top), float(yaw)

    def table_to_world(self, uv):
        """Table-frame xy offset(s) -> world xy."""
        c, s = np.cos(self.table_yaw), np.sin(self.table_yaw)
        return self.table_center + np.asarray(uv) @ np.array([[c, s], [-s, c]])

    def world_to_table(self, xy):
        c, s = np.cos(self.table_yaw), np.sin(self.table_yaw)
        return (np.asarray(xy) - self.table_center) @ np.array([[c, -s], [s, c]])

    def _sample_table(self, rng):
        """Random table pose in front of the robot: (centre xy, yaw). The nearest edge is table_gap beyond front_x."""
        sc = self.cfg.scene
        yaw = float(rng.uniform(*sc.table_yaw))
        hx, hy = self.table_half
        reach_back = hx * abs(np.cos(yaw)) + hy * abs(np.sin(yaw))   # centre -> nearest edge/corner along x
        return np.array([self.front_x + rng.uniform(*sc.table_gap) + reach_back, rng.uniform(*sc.table_y)]), yaw

    def _table_collides(self, clearance=None):
        """True if the table top comes within `clearance` (default SceneConfig.table_clearance) of any robot geom.

        Uses mj_geomDistance rather than contacts: the table and base_link are both world-welded, and MuJoCo never
        generates contacts between two world-welded bodies. Note base_link is a mesh, i.e. measured as its convex hull.
        """
        m, d = self.model, self.data
        clr = self.cfg.scene.table_clearance if clearance is None else clearance
        mujoco.mj_kinematics(m, d)
        for g in self.robot_geoms:
            if mujoco.mj_geomDistance(m, d, self.table_geom, int(g), clr + 1e-3, None) < clr:
                return True
        return False

    def _shelf_row(self, rng, active, bimanual):
        """{name: (world pos of base centre, yaw)} for every utensil: a row along the shelf (y) in shuffled order
        with random spacing. Bimanual episodes put the active utensil in a middle slot (both arms need small |y|)."""
        sc = self.cfg.scene
        cx, cy, z, hx, hy = self.shelf
        order = [self.object_names[i] for i in rng.permutation(len(self.object_names))]
        if bimanual and len(order) > 2:
            order.remove(active)
            order.insert(int(rng.integers(1, len(order))), active)
        half_w = np.array([self.specs[n].footprint_y for n in order])
        slack = 2 * hy - 2 * half_w.sum() - sc.shelf_min_gap * (len(order) + 1)
        if slack < 0:
            raise ValueError(f"Utensils {order} do not fit on the shelf ({2 * hy:.3f} m wide).")
        gaps = sc.shelf_min_gap + slack * rng.dirichlet(np.ones(len(order) + 1))
        poses, y = {}, cy - hy
        for n, w, g in zip(order, half_w, gaps):
            y += g + w
            spec = self.specs[n]
            if spec.grasp_style == "handle":         # handle points back, roughly at the arm on its side (re-aimed in reset)
                yaw = np.pi
                x_lo, x_hi = cx - hx + spec.extent, cx + hx - spec.radius
            else:
                yaw = rng.uniform(-np.pi, np.pi)
                x_lo, x_hi = cx - hx + spec.radius, cx + hx - spec.radius
            x = rng.uniform(x_lo, x_hi) if x_hi > x_lo else (x_lo + x_hi) / 2
            poses[n] = (np.array([x, y, z]), float(yaw))
            y += w
        return poses

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opt = options or {}
        m, d, sc, tc, rng = self.model, self.data, self.cfg.scene, self.cfg.task, self.np_random
        mujoco.mj_resetData(m, d)

        name = opt.get("object") or str(rng.choice(self.object_names))
        forced = opt.get("arm")
        bimanual = name in tc.bimanual_objects
        if forced is not None and (forced == "both") != bimanual:
            raise ValueError(f"{name} is always carried with {'both arms' if bimanual else 'one arm'}; "
                             f"arm={forced!r} is not allowed (see TaskConfig.bimanual_objects).")
        self._set_active(name, ("left", "right") if bimanual else ("left",))
        self._set_home(None if opt.get("_baseline") else rng)
        mujoco.mj_forward(m, d)
        self.reach_fallback = False
        if opt.get("_baseline"):
            return self._get_obs(), {}
        spec = self.spec

        # 1) all utensils in a shuffled row on the shelf; the active one must be reachable by the arm(s) used
        for attempt in range(100):
            poses = self._shelf_row(rng, name, bimanual)
            pos, yaw0 = poses[name]
            if bimanual:
                sides = ("left", "right")
            elif forced in SIDES:
                sides = (forced,)
            elif tc.arm_mode == "auto":
                sides = ("left",) if pos[1] >= 0 else ("right",)
            else:
                sides = (tc.arm_mode,)
            if spec.grasp_style == "handle":         # the handle must point at the arm that grasps it
                yaw0 = self.yaw_to_arm(pos, sides[0]) + rng.uniform(-0.15, 0.15)
                poses[name] = (pos, yaw0)
            if ((not bimanual or abs(pos[1]) <= tc.bimanual_y_max)
                    and self.reachable(self.grasp_plan(pos, sides, yaw0), sides)):
                break
        else:
            self.reach_fallback = True
        self._set_active(name, sides)

        # 2) destination table in front of the robot (x, y, yaw; fixed height) + a goal on it, reachable by the
        #    same arm(s); table poses that touch the robot are rejected
        goal_ok, top = False, float(sc.table_z)
        for attempt in range(60):
            if "table_pos" in opt:
                tp = opt["table_pos"]
                c, yaw = np.array(tp[:2], dtype=float), float(opt.get("table_yaw", 0.0))
                top = float(tp[2]) if len(tp) > 2 else top
            else:
                c, yaw = self._sample_table(rng)
            self._place_table(c, top, yaw)
            if "table_pos" not in opt and self._table_collides():
                continue
            half = np.maximum(self.table_half - spec.extent - sc.goal_margin, 0.0)
            for _ in range(20):
                gxy = (np.array(opt["goal_pos"][:2]) if "goal_pos" in opt
                       else self.table_to_world(rng.uniform(-1, 1, 2) * half))
                goal = np.array([gxy[0], gxy[1], top])
                # the object keeps its yaw while carried, except a mug: its handle stays aimed at the arm
                gyaw = self.yaw_to_arm(goal, sides[0]) if spec.grasp_style == "handle" else yaw0
                if "goal_pos" in opt or "table_pos" in opt or self.reachable(self.grasp_plan(goal, sides, gyaw), sides):
                    goal_ok = True
                    break
            if goal_ok:
                break
        if not goal_ok:
            self.reach_fallback = True
            warnings.warn("No reachable table pose found; using the last candidate. Shift/widen SceneConfig.table_* "
                          "(or table_z) towards the arms' reach.", RuntimeWarning, stacklevel=2)
            goal = np.array([*self.table_center, self.place_z])
        self.goal_pos = goal
        self.src_z = float(pos[2])
        self.fail_z = min(self.place_z, self.src_z) - 0.25

        # 3) physics randomisation + place the utensils
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
        for n, (p, oyaw) in poses.items():
            qa = self.obj_qadr[n]
            d.qpos[qa:qa + 7] = [p[0], p[1], p[2] + 0.002, *_yaw_quat(oyaw)]

        mujoco.mj_forward(m, d)
        for _ in range(30):
            mujoco.mj_step(m, d)
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
