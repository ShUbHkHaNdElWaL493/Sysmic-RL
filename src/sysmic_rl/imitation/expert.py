"""Privileged scripted expert (Jacobian IK + phase machine) used to generate demonstrations.

It outputs actions in the *same* 13-D normalised joint-delta space as the policy,
so labels can be used directly for behaviour cloning / DAgger. It reads simulator
state (privileged), the policy only sees the observation vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

(PRE, APPROACH, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT, DONE) = range(9)
PHASE_NAMES = ["pre", "approach", "close", "lift", "transport", "lower", "release", "retreat", "done"]


@dataclass
class ExpertConfig:
    pre_dist: float = 0.07          # stand-off along the approach axis
    lift_height: float = 0.08
    pos_tol: float = 0.012
    axis_tol: float = 0.25
    damping: float = 0.05
    pos_gain: float = 0.3
    rot_gain: float = 0.3
    rot_weight: float = 0.5
    lift_col_scale: float = 0.5     # discourage using the lift joint when the arm can reach
    close_steps: int = 15
    release_steps: int = 12
    phase_timeout: int = 150
    side_height_frac: float = 0.6
    top_depth: float = 0.006
    regrasp_after: int = 6


class ScriptedExpert:
    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self.reset()

    def reset(self):
        self.phase, self.t, self.lost = PRE, 0, 0
        self.lift_z = None

    # --- geometry of the grasp for the current object
    def _grasp_targets(self, env):
        spec, c = env.spec, self.cfg
        o = env.object_pos()
        if spec.grasp_style == "top":
            g = o + np.array([-spec.rim_radius, 0.0, spec.height - c.top_depth])
            axis, pre = np.array([0.0, 0.0, -1.0]), g + np.array([0, 0, c.pre_dist])
        else:
            g = o + np.array([0.0, 0.0, c.side_height_frac * spec.height])
            axis, pre = np.array([1.0, 0.0, 0.0]), g + np.array([-c.pre_dist, 0, 0])
        return g, pre, axis

    def _goto(self, env, pt_target, axis_target, grip_q):
        c, m, d = self.cfg, env.model, env.data
        side = env.side
        pt = env.grasp_point()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, pt, env.finger_bid[side])
        cols = np.r_[env.lift_dof, env.arm_dofs(side)[:5]]
        a = env.tool_axis_world()
        proj = np.eye(3) - np.outer(a, a)
        J = np.vstack([jp[:, cols], c.rot_weight * (proj @ jr[:, cols])])
        e = np.r_[c.pos_gain * (pt_target - pt), c.rot_weight * c.rot_gain * np.cross(a, axis_target)]
        cs = np.ones(6)
        cs[0] = c.lift_col_scale
        Jw = J * cs
        dq = cs * (Jw.T @ np.linalg.solve(Jw @ Jw.T + c.damping ** 2 * np.eye(6), e))

        cur, tgt = env.joint_pos(), env.target
        sl = env.sl[side]
        idx = np.r_[0, np.arange(sl.start, sl.start + 5)]
        act = np.zeros(13)
        act[idx] = (cur[idx] + dq - tgt[idx]) / env.act_scale[idx]
        g = sl.stop - 1
        act[g] = (grip_q - tgt[g]) / env.act_scale[g]
        return np.clip(act, -1, 1), float(np.linalg.norm(pt_target - pt)), float(np.arccos(np.clip(a @ axis_target, -1, 1)))

    def _advance(self, nxt):
        self.phase, self.t = nxt, 0

    def act(self, env) -> np.ndarray:
        c, tc = self.cfg, env.cfg.task
        self.t += 1
        gq_open, gq_close = tc.gripper_open_q, tc.gripper_close_q
        g, pre, axis = self._grasp_targets(env)
        pt, o = env.grasp_point(), env.object_pos()
        grasping = env.is_grasping()
        timeout = self.t > c.phase_timeout

        if self.phase in (LIFT, TRANSPORT, LOWER):  # recovery when the object is lost
            self.lost = 0 if grasping else self.lost + 1
            if self.lost >= c.regrasp_after:
                self.lost = 0
                self._advance(PRE)

        ph = self.phase
        if ph == PRE:
            a, ep, er = self._goto(env, pre, axis, gq_open)
            if (ep < c.pos_tol * 1.5 and er < c.axis_tol) or timeout:
                self._advance(APPROACH)
        elif ph == APPROACH:
            a, ep, er = self._goto(env, g, axis, gq_open)
            if ep < c.pos_tol or timeout:
                self._advance(CLOSE)
        elif ph == CLOSE:
            a, _, _ = self._goto(env, g, axis, gq_close)
            if (self.t >= c.close_steps and grasping) or self.t > c.close_steps * 4:
                self.lift_z = pt[2] + c.lift_height
                self._advance(LIFT)
        elif ph == LIFT:
            tgt = np.array([pt[0], pt[1], self.lift_z])
            a, ep, _ = self._goto(env, tgt, axis, gq_close)
            if ep < c.pos_tol or timeout:
                self._advance(TRANSPORT)
        elif ph == TRANSPORT:
            delta = env.goal_pos[:2] - o[:2]
            tgt = np.array([pt[0] + delta[0], pt[1] + delta[1], self.lift_z])
            a, _, _ = self._goto(env, tgt, axis, gq_close)
            if np.linalg.norm(delta) < c.pos_tol or timeout:
                self._advance(LOWER)
        elif ph == LOWER:
            delta = env.goal_pos[:2] - o[:2]
            dz = (env.goal_pos[2] + 0.003) - o[2]
            tgt = pt + np.array([delta[0], delta[1], dz])
            a, _, _ = self._goto(env, tgt, axis, gq_close)
            if abs(dz) < 0.006 or timeout:
                self._advance(RELEASE)
        elif ph == RELEASE:
            a, _, _ = self._goto(env, pt, axis, gq_open)
            if self.t >= c.release_steps:
                self._advance(RETREAT)
        elif ph == RETREAT:
            back = np.array([0, 0, 0.08]) if env.spec.grasp_style == "top" else np.array([-0.08, 0, 0])
            a, ep, _ = self._goto(env, pt + back * 0.5, axis, gq_open)
            if self.t > 25:
                self._advance(DONE)
        else:
            a = np.zeros(13)
            gi = env.sl[env.side].stop - 1
            a[gi] = np.clip((gq_open - env.target[gi]) / env.act_scale[gi], -1, 1)
        return (a * env.action_mask).astype(np.float32)
