"""Privileged scripted expert (Jacobian IK + phase machine) used to generate demonstrations.

Works with one arm or both: all active arms are solved TOGETHER in one damped-least-squares IK
(they share the lift joint), and share one phase machine so both hands close/lift/release in sync.
It outputs actions in the same 13-D normalised joint-delta space as the policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

(SEED, PRE, APPROACH, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT, DONE) = range(10)
PHASE_NAMES = ["seed", "pre", "approach", "close", "lift", "transport", "lower", "release", "retreat", "done"]


@dataclass
class ExpertConfig:
    pre_dist: float = 0.07          # stand-off along the approach axis
    lift_height: float = 0.08
    pos_tol: float = 0.012
    axis_tol: float = 0.25
    damping: float = 0.05
    pos_gain: float = 0.3
    rot_gain: float = 0.3
    rot_weight: float = 0.15
    lift_col_scale: float = 0.5     # discourage using the lift joint when the arms can reach
    close_steps: int = 15
    release_steps: int = 12
    phase_timeout: int = 150
    regrasp_after: int = 6
    retreat_dist: float = 0.04
    seed_tol: float = 0.06          # rad: joint-space seed move is done when all joints are this close


class ScriptedExpert:
    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self.reset()

    def reset(self):
        self.phase, self.t, self.lost = SEED, 0, 0
        self.lift_z = {}
        self.seed = None

    def _goto(self, env, goals, lock_lift=False):
        """goals: {side: (target_point, tool_axis, gripper_q)} -> (action[13], max_pos_err, max_axis_err)."""
        c, m, d = self.cfg, env.model, env.data
        sides = list(goals)
        cols = [env.lift_dof] + [int(x) for s in sides for x in env.arm_dofs(s)[:5]]
        J_rows, e_rows, pos_err, ax_err = [], [], 0.0, 0.0
        for s in sides:
            tp, tax, _ = goals[s]
            pt, a = env.grasp_point(s), env.tool_axis_world(s)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, pt, env.finger_bid[s])
            proj = np.eye(3) - np.outer(a, a)   # ignore rotation about the tool axis (wrist roll stays free)
            J_rows += [jp[:, cols], c.rot_weight * (proj @ jr[:, cols])]
            e_rows += [c.pos_gain * (tp - pt), c.rot_weight * c.rot_gain * np.cross(a, tax)]
            pos_err = max(pos_err, float(np.linalg.norm(tp - pt)))
            ax_err = max(ax_err, float(np.arccos(np.clip(a @ tax, -1, 1))))
        J, e = np.vstack(J_rows), np.concatenate(e_rows)
        cs = np.ones(len(cols))
        cs[0] = 1e-3 if lock_lift else c.lift_col_scale   # lock: the target rides on the lift, IK must not use it
        Jw = J * cs
        dq = cs * (Jw.T @ np.linalg.solve(Jw @ Jw.T + c.damping ** 2 * np.eye(len(e)), e))

        cur, tgt, sc = env.joint_pos(), env.target, env.act_scale
        act = np.zeros(13)
        act[0] = (cur[0] + dq[0] - tgt[0]) / sc[0]
        k = 1
        for s in sides:
            sl = env.sl[s]
            idx = np.arange(sl.start, sl.start + 5)
            act[idx] = (cur[idx] + dq[k:k + 5] - tgt[idx]) / sc[idx]
            g = sl.stop - 1
            act[g] = (goals[s][2] - tgt[g]) / sc[g]
            k += 5
        return np.clip(act, -1, 1), pos_err, ax_err

    def _advance(self, nxt):
        self.phase, self.t = nxt, 0

    def act(self, env) -> np.ndarray:
        c, tc = self.cfg, env.cfg.task
        self.t += 1
        gq_open, gq_close = tc.gripper_open_q, tc.gripper_close_q
        sides = env.sides
        plan = env.grasp_plan()
        pts = {s: env.grasp_point(s) for s in sides}
        o = env.object_pos()
        grasping = env.is_grasping()
        timeout = self.t > c.phase_timeout
        G = lambda tgt, gq: {s: (tgt[s], plan[s][1], gq) for s in sides}
        lock = env.surface_moves and self.phase <= CLOSE   # object still sits on the lift-mounted surface

        if self.phase in (LIFT, TRANSPORT, LOWER):  # recovery when the object is lost
            self.lost = 0 if grasping else self.lost + 1
            if self.lost >= c.regrasp_after:
                self.lost = 0
                self._advance(PRE)   # arms are already close: skip the seed move

        ph = self.phase
        if ph == SEED:
            # Global-IK seed: jump (in joint space) near a configuration that already has the right pose.
            if self.seed is None:
                pre = {s: (plan[s][0] - plan[s][1] * c.pre_dist, plan[s][1]) for s in sides}
                self.seed = env.seed_config(pre, source=True) or "none"
            a = np.zeros(13)
            if self.seed == "none":
                self._advance(PRE)
            else:
                lift, q5, _ = self.seed
                cur, tgt, sc = env.joint_pos(), env.target, env.act_scale
                a[0] = (lift - tgt[0]) / sc[0]
                err = abs(lift - cur[0]) * 10
                for s in sides:
                    sl = env.sl[s]
                    idx = np.arange(sl.start, sl.start + 5)
                    a[idx] = (q5[s] - tgt[idx]) / sc[idx]
                    a[sl.stop - 1] = (gq_open - tgt[sl.stop - 1]) / sc[sl.stop - 1]
                    err = max(err, float(np.max(np.abs(q5[s] - cur[idx]))))
                a = np.clip(a, -1, 1)
                if err < c.seed_tol or timeout:
                    self._advance(PRE)
        elif ph == PRE:
            a, ep, er = self._goto(env, G({s: plan[s][0] - plan[s][1] * c.pre_dist for s in sides}, gq_open), lock)
            if (ep < c.pos_tol * 1.5 and er < c.axis_tol) or timeout:
                self._advance(APPROACH)
        elif ph == APPROACH:
            a, ep, _ = self._goto(env, G({s: plan[s][0] for s in sides}, gq_open), lock)
            if ep < c.pos_tol or timeout:
                self._advance(CLOSE)
        elif ph == CLOSE:
            a, _, _ = self._goto(env, G({s: plan[s][0] for s in sides}, gq_close), lock)
            if (self.t >= c.close_steps and grasping) or self.t > c.close_steps * 4:
                self.lift_z = {s: pts[s][2] + c.lift_height for s in sides}
                self._advance(LIFT)
        elif ph == LIFT:
            a, ep, _ = self._goto(env, G({s: np.array([pts[s][0], pts[s][1], self.lift_z[s]]) for s in sides}, gq_close))
            if ep < c.pos_tol or timeout:
                self._advance(TRANSPORT)
        elif ph == TRANSPORT:
            delta = env.goal_pos[:2] - o[:2]
            a, _, _ = self._goto(env, G({s: np.array([pts[s][0] + delta[0], pts[s][1] + delta[1], self.lift_z[s]])
                                         for s in sides}, gq_close))
            if np.linalg.norm(delta) < c.pos_tol or timeout:
                self._advance(LOWER)
        elif ph == LOWER:
            delta = env.goal_pos[:2] - o[:2]
            dz = (env.goal_pos[2] + 0.003) - o[2]
            a, _, _ = self._goto(env, G({s: pts[s] + np.array([delta[0], delta[1], dz]) for s in sides}, gq_close))
            if abs(dz) < 0.006 or timeout:
                self._advance(RELEASE)
        elif ph == RELEASE:
            a, _, _ = self._goto(env, G(pts, gq_open))
            if self.t >= c.release_steps:
                self._advance(RETREAT)
        elif ph == RETREAT:
            a, _, _ = self._goto(env, G({s: pts[s] - plan[s][1] * c.retreat_dist for s in sides}, gq_open))
            if self.t > 25:
                self._advance(DONE)
        else:
            a = np.zeros(13)
            for s in sides:
                gi = env.sl[s].stop - 1
                a[gi] = np.clip((gq_open - env.target[gi]) / env.act_scale[gi], -1, 1)
        return (a * env.action_mask).astype(np.float32)
