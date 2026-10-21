"""Premaid AI get-up task: from lying on its back, front or side to the walking policy's crouch.

Shares the robot, servo model (delays, slew limit, randomization) and the 86-value observation
layout with PremaidWalkEnv (commands and gait clock are always zero here), so the same deploy code
can run either policy and hand over from get-up to walking.

Differences from walking:
  * starts from a pool of settled lying poses (robot dropped limp in random orientations)
  * full self-collision (arms vs torso, legs vs torso...) since the body folds up while getting up
  * large action range: target = crouch pose + 1.0 * action (clipped to joint limits)
  * no falling termination, no pushes, no debris
  * rewards: get upright, get the torso up, then settle into the crouch with both feet down;
    penalties keep motions smooth and servos out of torque saturation (real servos overheat)
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from premaid_env import NUM_ACTIONS, PremaidWalkEnv, WalkCfg


@dataclasses.dataclass
class GetUpCfg(WalkCfg):
    episode_length_s: float = 10.0
    action_scale_leg: float = 1.0
    action_scale_upper: float = 1.0
    cmd_stand_prob: float = 1.0  # commands always zero
    debris: bool = False
    push_interval_s: tuple = (1e9, 1e9)
    self_collision: bool = True

    # lying-pose pool
    pool_size: int = 48
    pool_prone_frac: float = 0.4  # face down; the rest face up or on a side
    init_noise: float = 0.05  # rad added to pool joint angles

    # rewards (x dt)
    w_upright: float = 2.0  # 0..1 from lying flat to vertical torso
    w_height: float = 2.0  # HEAD height / standing head height (hip height alone rewards kneeling head-down)
    w_hip_height: float = 2.0  # hip (base) height / standing height, only counts while upright
    w_feet_down: float = 1.0  # while upright: both soles on the floor (not kneeling / sitting)
    w_leg_pose: float = 1.0  # while upright: legs folding toward the crouch shape (feet under the body)
    w_stand_pose: float = 2.0  # once up: joints near the crouch the walker starts from
    w_stand_feet: float = 1.0  # once up: both feet on the floor
    w_stand_still: float = 1.0  # once up: torso not moving
    w_action_rate: float = -0.005
    w_torque: float = -0.01
    w_joint_vel: float = -0.0005
    w_ang_vel: float = -0.02
    w_saturation: float = -0.2  # fraction of servos at their torque limit
    w_joint_limit: float = -2.0


class PremaidGetUpEnv(PremaidWalkEnv):
    cfg: GetUpCfg

    def __init__(self, cfg: GetUpCfg | None = None, seed: int = 0, **kw):
        cfg = cfg or GetUpCfg()
        self._pool = None
        super().__init__(cfg, seed=seed, **kw)
        m = self.model
        self.robot_geoms = np.array([
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] != 0 and m.body_mocapid[m.geom_bodyid[g]] < 0 and (m.geom_contype[g] or m.geom_conaffinity[g])
        ])
        if cfg.self_collision:
            m.geom_contype[self.robot_geoms] |= 4
            m.geom_conaffinity[self.robot_geoms] |= 4
        self.force_lim = m.actuator_forcerange[:, 1].copy()
        self.head_id = m.body("head").id
        from premaid_env import LEG_JOINTS, SERVO_ORDER

        self.leg_idx = np.array([SERVO_ORDER.index(n) for n in LEG_JOINTS])
        self._place_default(0.0)
        self.stand_head_z = float(self.data.xpos[self.head_id][2])
        self._pool = self._make_pool()
        self.reset()

    # ------------------------------------------------------------------ lying poses
    def _make_pool(self):
        """Drop the robot limp from random orientations and keep the settled states."""
        c, m, r = self.cfg, self.model, self.rng
        d = mujoco.MjData(m)
        gain, bias = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, :3].copy()
        m.actuator_gainprm[:, 0] = 0.0
        m.actuator_biasprm[:, :3] = 0.0
        damping = m.dof_damping.copy()
        m.dof_damping[6:] = 0.02
        pool, attempts = [], 0
        try:
            while len(pool) < c.pool_size and attempts < 20 * c.pool_size:
                attempts += 1
                mujoco.mj_resetData(m, d)
                kind = r.random()
                if kind < c.pool_prone_frac:
                    pitch = np.pi / 2  # face down
                elif kind < 0.85:
                    pitch = -np.pi / 2  # face up
                else:
                    pitch = r.choice([-1, 1]) * r.uniform(0.3, 1.2)
                roll = r.uniform(-0.5, 0.5) + (r.choice([-1, 1]) * np.pi / 2 if kind >= 0.85 else 0.0)
                yaw = r.uniform(-np.pi, np.pi)
                d.qpos[3:7] = _euler_to_quat(roll, pitch, yaw)
                lo, hi = self.jnt_lo, self.jnt_hi
                d.qpos[self.qadr] = np.clip(self.default_q + r.normal(0, 0.35, NUM_ACTIONS), lo, hi)
                d.qpos[2] = 0.35
                for _ in range(int(2.0 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                if np.abs(d.qvel).max() < 0.5 and d.qpos[2] < 0.15:
                    pool.append((d.qpos.copy(), d.qvel.copy() * 0.0))
        finally:
            m.actuator_gainprm[:, 0] = gain
            m.actuator_biasprm[:, :3] = bias
            m.dof_damping[:] = damping
        return pool

    # ------------------------------------------------------------------ API
    def reset(self):
        if self._pool is None:  # called from the base constructor before the pool exists
            return super().reset()
        c, m, d, r = self.cfg, self.model, self.data, self.rng
        self._randomize()
        if not hasattr(self, "friction"):
            self.friction = float(m.geom_friction[self.floor_id, 0])
        mujoco.mj_resetData(m, d)
        qpos, qvel = self._pool[r.integers(len(self._pool))]
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        yaw = r.uniform(-np.pi, np.pi)  # spin the whole pose about the vertical axis
        d.qpos[3:7] = _quat_mul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), d.qpos[3:7])
        d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + r.normal(0, c.init_noise, NUM_ACTIONS), self.jnt_lo, self.jnt_hi)
        d.qpos[2] += 0.005
        self._place_debris(np.arange(len(self.debris_ids)), (0.0, 0.0))
        mujoco.mj_forward(m, d)
        self._reset_servos()  # servos start by holding the pose they're in
        d.ctrl[:] = self.servo_ref

        self.step_count = 0
        self.last_action = np.clip((d.qpos[self.qadr] - self.default_q) / self.action_scale, -c.action_clip, c.action_clip)
        self.last_dq = np.zeros(NUM_ACTIONS)
        self.foot_pos_prev = d.site_xpos[self.foot_site_ids].copy()
        self.next_push = 1e9
        self.push_flag = 0.0
        self.commands[:] = 0.0
        self.ep_sums = {}
        self.stood_at = None
        return self._observe()

    def _sample_commands(self):
        self.commands[:] = 0.0

    def _update_curriculum(self, terminated_early: bool):
        pass

    def _reward(self, action, torque_sq):
        c, d, dt = self.cfg, self.data, self.dt
        R = d.xmat[self.base_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[self._sid["gyro"]: self._sid["gyro"] + 3]
        q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
        contact, _ = self._foot_contacts()

        upright = float(np.clip(-grav[2], 0.0, 1.0))
        height = float(np.clip(d.qpos[2] / self.stand_height, 0.0, 1.1))
        head = float(np.clip(d.xpos[self.head_id][2] / self.stand_head_z, 0.0, 1.1))
        standing = upright > 0.9 and height > 0.85
        t = {}
        t["upright"] = c.w_upright * upright
        t["height"] = c.w_height * min(head, 1.0)
        t["hip_height"] = c.w_hip_height * min(height, 1.0) * upright
        t["feet_down"] = c.w_feet_down * float(contact.all()) * upright
        leg = self.leg_idx
        t["leg_pose"] = c.w_leg_pose * np.exp(-np.sum((q[leg] - self.default_q[leg]) ** 2) / 0.5) * upright
        if standing:
            t["stand_pose"] = c.w_stand_pose * np.exp(-np.sum((q - self.default_q) ** 2) / 0.5)
            t["stand_feet"] = c.w_stand_feet * float(contact.all())
            t["stand_still"] = c.w_stand_still * np.exp(-np.sum(d.qvel[:6] ** 2) / 0.1)
        else:
            t["stand_pose"] = t["stand_feet"] = t["stand_still"] = 0.0
        t["action_rate"] = c.w_action_rate * np.sum((action - self.last_action) ** 2)
        t["torque"] = c.w_torque * torque_sq
        t["joint_vel"] = c.w_joint_vel * np.sum(dq**2)
        t["ang_vel"] = c.w_ang_vel * np.sum(gyro**2)
        t["saturation"] = c.w_saturation * float(np.mean(np.abs(d.actuator_force) > 0.95 * self.force_lim))
        span = self.jnt_hi - self.jnt_lo
        lo, hi = self.jnt_lo + 0.05 * span, self.jnt_hi - 0.05 * span
        t["joint_limit"] = c.w_joint_limit * np.sum(np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None))
        t = {k: float(v) * dt for k, v in t.items()}

        if standing and self.stood_at is None:
            self.stood_at = self.step_count * dt
        self._standing_now = standing
        return float(sum(t.values())), t, False

    def episode_summary(self):
        out = super().episode_summary()
        out.pop("Curriculum/debris_level", None)
        out["GetUp/standing_at_end"] = float(getattr(self, "_standing_now", False))
        out["GetUp/time_to_stand_s"] = self.stood_at if self.stood_at is not None else self.cfg.episode_length_s
        return out


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
