"""Pick-and-place Gymnasium environment for the Sysmic bimanual robot.

Keeps the 13-D joint-delta action interface of SysmicV2Env (lift + 2x[5 joints + gripper])
so a reaching policy can be used as initialisation.  One arm is active per episode.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from .config import ARMS, PickPlaceConfig
from .objects import make_object_specs
from .scene_builder import build_scene_xml

SIDES = ("left", "right")


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

        # episode state
        self.step_count = 0
        self.side = "left"
        self.obj_name = self.object_names[0]
        self.goal_pos = np.array([0.3, 0.0, sc.table_top_z], dtype=np.float64)
        self.target = np.zeros(13)
        self.smoothed = np.zeros(13, dtype=np.float32)
        self.ever_grasped = False
        self.hold = 0
        self.rest_com_z = 0.0
        self._set_active(self.obj_name, "left")
        self._set_home()
        mujoco.mj_forward(self.model, self.data)
        self._baseline_pairs = set()
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
        self.table_geom = m.geom("table_top").id

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

    def _set_active(self, obj_name, side):
        self.obj_name, self.side = obj_name, side
        self.active_idx = self.object_names.index(obj_name)
        self.spec = self.specs[obj_name]
        self.action_mask = np.zeros(13, dtype=np.float32)
        self.action_mask[0] = 1.0
        self.action_mask[self.sl[side]] = 1.0

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
        R = self.data.xmat[b].reshape(3, 3)
        return self.data.xpos[b] + R @ np.array(ARMS[side].grasp_offset)

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

    # ---------------------------------------------------------------- contacts
    def _grasp_flags(self):
        d = self.data
        n = d.ncon
        if n == 0:
            return False, False
        g1, g2 = d.contact.geom[:n, 0], d.contact.geom[:n, 1]
        touching = d.contact.dist[:n] < 0.001
        o1, o2 = self.geom_obj[g1] == self.active_idx, self.geom_obj[g2] == self.active_idx
        f, j = self.geom_finger[self.side], self.geom_jaw[self.side]
        fin = np.any(touching & ((o1 & f[g2]) | (o2 & f[g1])))
        jaw = np.any(touching & ((o1 & j[g2]) | (o2 & j[g1])))
        return bool(fin), bool(jaw)

    def is_grasping(self):
        f, j = self._grasp_flags()
        return f and j

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
            if (g1 == self.table_geom and self.geom_tip[g2]) or (g2 == self.table_geom and self.geom_tip[g1]):
                continue
            if ignore_baseline and frozenset((int(gb[g1]), int(gb[g2]))) in self._baseline_pairs:
                continue
            out.append(float(dist[i]))
        return out

    def _compute_baseline_contacts(self):
        """Contacts present at the home pose are mesh-overlap artefacts: don't penalise them."""
        self.reset(seed=0, options={"_baseline": True})
        gb = self.model.geom_bodyid
        d = self.data
        pairs = set()
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i, 0]), int(d.contact.geom[i, 1])
            if self.geom_arm[g1] > 0 or self.geom_arm[g2] > 0:
                if self.geom_obj[g1] < 0 and self.geom_obj[g2] < 0:
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
        arm1h = np.array([self.side == "left", self.side == "right"], dtype=np.float64)
        fin, jaw = self._grasp_flags()
        return [
            ("q_norm", qn), ("qvel", qv),
            ("grasp_pos", np.r_[gp["left"], gp["right"]]),
            ("grasp_quat", np.r_[quat["left"], quat["right"]]),
            ("prev_action", self.smoothed),
            ("obj_pos", d.xpos[b]), ("obj_quat", d.xquat[b]),
            ("obj_linvel", d.cvel[b][3:]), ("obj_angvel", d.cvel[b][:3]),
            ("obj_type", onehot),
            ("obj_rel_ee", np.r_[com - gp["left"], com - gp["right"]]),
            ("goal_pos", self.goal_pos), ("goal_rel_obj", self.goal_pos - d.xpos[b]),
            ("active_arm", arm1h), ("contacts", np.array([fin, jaw], dtype=np.float64)),
        ]

    def _get_obs(self):
        return np.concatenate([v for _, v in self._obs_parts()]).astype(np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opt = options or {}
        m, d, sc, tc, rng = self.model, self.data, self.cfg.scene, self.cfg.task, self.np_random
        mujoco.mj_resetData(m, d)

        name = opt.get("object") or str(rng.choice(self.object_names))
        side = opt.get("arm") or tc.arm_mode
        if side == "auto":
            side = SIDES[int(rng.integers(2))]
        self._set_active(name, side)
        self._set_home(rng)

        ylo, yhi = sc.spawn_y
        yr = (max(ylo, 0.0), yhi) if side == "left" else (ylo, min(yhi, 0.0))
        xy = np.array(opt["object_pos"][:2]) if "object_pos" in opt else np.array(
            [rng.uniform(*sc.spawn_x), rng.uniform(*yr)])
        goal_xy = None
        if "goal_pos" in opt:
            goal_xy = np.array(opt["goal_pos"][:2])
        else:
            for _ in range(100):
                goal_xy = np.array([rng.uniform(*sc.spawn_x), rng.uniform(*yr)])
                if np.linalg.norm(goal_xy - xy) >= tc.min_goal_dist:
                    break
        self.goal_pos = np.array([goal_xy[0], goal_xy[1], sc.table_top_z])

        yaw = float(opt.get("object_yaw", rng.uniform(-0.3, 0.3) if self.spec.grasp_style == "side"
                            else rng.uniform(-np.pi, np.pi)))
        qa = self.obj_qadr[name]
        d.qpos[qa:qa + 7] = [xy[0], xy[1], sc.table_top_z + 0.002, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        m.geom_friction[:] = self._fric0
        m.body_mass[:] = self._mass0
        m.body_inertia[:] = self._inertia0
        if tc.randomize_physics and not opt.get("_baseline"):
            b = self.obj_body[name]
            gsel = self.geom_obj == self.active_idx
            m.geom_friction[gsel, 0] = self._fric0[gsel, 0] * rng.uniform(*tc.friction_scale)
            s = rng.uniform(*tc.mass_scale)
            m.body_mass[b] = self._mass0[b] * s
            m.body_inertia[b] = self._inertia0[b] * s

        self._finger_geoms_ready = True
        mujoco.mj_forward(m, d)
        for _ in range(30):
            mujoco.mj_step(m, d)
        if self._goal_mocap >= 0:
            d.mocap_pos[self._goal_mocap] = self.goal_pos + np.array([0, 0, 0.03])
        self.step_count, self.hold, self.ever_grasped = 0, 0, False
        self.smoothed = np.zeros(13, dtype=np.float32)
        self.rest_com_z = float(self.object_com()[2])
        mujoco.mj_forward(m, d)
        return self._get_obs(), {"object": name, "arm": side}

    # ------------------------------------------------------------------- step
    def step(self, action):
        tc, rw, sc = self.cfg.task, self.cfg.reward, self.cfg.scene
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
        tc, rw, sc = self.cfg.task, self.cfg.reward, self.cfg.scene
        d = self.data
        com, pos = self.object_com(), self.object_pos()
        gp = self.grasp_point()
        fin, jaw = self._grasp_flags()
        grasped = fin and jaw
        self.ever_grasped |= grasped
        d_reach = float(np.linalg.norm(gp - com))
        lift_h = com[2] - self.rest_com_z
        lifted = lift_h > 0.015
        e_xy = float(np.linalg.norm(pos[:2] - self.goal_pos[:2]))
        e_z = float(abs(pos[2] - self.goal_pos[2]))
        e_goal = float(np.linalg.norm(pos - self.goal_pos))
        tilt, speed = self.object_tilt(), self.object_speed()
        released = not (fin or jaw)
        placed = (e_xy < tc.place_tol_xy and e_z < tc.place_tol_z
                  and tilt < tc.place_max_tilt and speed < tc.place_max_speed)
        self.hold = self.hold + 1 if (placed and released) else 0
        success = self.hold >= tc.hold_steps
        failed = bool(pos[2] < sc.table_top_z - 0.05 or tilt > tc.fail_tilt)

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
            "success": bool(success), "failed": failed, "grasped": bool(grasped), "lifted": bool(lifted),
            "placed": bool(placed), "released": bool(released), "err_reach": d_reach, "err_goal_xy": e_xy,
            "tilt": tilt, "n_bad_contacts": len(bad), "object": self.obj_name, "arm": self.side,
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [0.25, 0, 0.8]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.3, 145, -25
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
