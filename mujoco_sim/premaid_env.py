"""Premaid AI velocity-command walking task on plain MuJoCo (CPU).

One `PremaidWalkEnv` = one robot in its own MjModel/MjData (so domain
randomization can change model parameters per robot). Batching across CPU
cores lives in `vec_env.py`.

Actor observation (86, all things the real robot can measure):
    gyro (3)  [BNO055 angular velocity, body frame]
    projected gravity (3)  [from BNO055 orientation]
    command vx, vy, yaw rate (3)
    joint position - default (25)   [servo order, see SERVO_ORDER]
    joint velocity (25)
    previous action (25)
    gait clock sin/cos (2)

Critic state (96) = actor observation without noise + privileged terms:
    base linear velocity (3), base height (1), foot contacts (2), foot heights (2),
    friction coefficient (1), push in progress (1)
"""

from __future__ import annotations

import dataclasses
import os

import mujoco
import numpy as np

ASSET_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "premaidai.xml")

# Servo-ID order (matches the actuator order in premaidai.xml and the real robot's servo map).
SERVO_ORDER = [
    "r_shy", "head_ny", "l_shy", "head_nz", "r_hpz", "head_nx",
    "l_hpz", "r_shx", "r_hpx", "l_shx", "l_hpx", "r_ely",
    "r_hpy", "l_ely", "l_hpy", "r_elx", "r_kny", "l_elx",
    "l_kny", "r_wry", "r_aky", "l_wry", "l_aky", "r_akx",
    "l_akx",
]

# Slightly crouched standing pose, arms down along the body (radians).
DEFAULT_POSE = {
    "r_hpy": -0.35, "l_hpy": -0.35,
    "r_kny": 0.70, "l_kny": 0.70,
    "r_aky": -0.35, "l_aky": -0.35,
    "r_shx": 1.35, "l_shx": -1.35,
}

LEG_JOINTS = [n for n in SERVO_ORDER if n[2:4] in ("hp", "kn", "ak")]
UPPER_JOINTS = [n for n in SERVO_ORDER if n not in LEG_JOINTS]

NUM_ACTIONS = 25
NUM_OBS = 3 + 3 + 3 + 25 + 25 + 25 + 2
NUM_STATES = NUM_OBS + 3 + 1 + 2 + 2 + 1 + 1


@dataclasses.dataclass
class WalkCfg:
    # timing
    control_dt: float = 0.02  # 50 Hz policy (sim runs at 200 Hz -> decimation 4)
    episode_length_s: float = 20.0
    gait_period: float = 0.5  # one full left+right stride cycle (s)

    # actions: target = default_pose + scale * action
    action_scale_leg: float = 0.30
    action_scale_upper: float = 0.20
    action_clip: float = 3.0

    # command ranges (this is a 0.38 m tall robot; ~0.15 m/s is a brisk walk)
    cmd_vx: tuple = (-0.10, 0.20)
    cmd_vy: tuple = (-0.08, 0.08)
    cmd_yaw: tuple = (-0.6, 0.6)
    cmd_resample_s: float = 5.0
    cmd_stand_prob: float = 0.15
    cmd_deadband: float = 0.03

    # termination
    min_base_height: float = 0.15
    max_tilt_cos: float = 0.6  # terminate when |gravity_xy| exceeds ~ sin(53 deg)

    # domain randomization
    randomize: bool = True
    friction_range: tuple = (0.4, 1.2)
    body_mass_scale: tuple = (0.9, 1.15)
    kp_scale: tuple = (0.8, 1.2)
    damping_scale: tuple = (0.7, 1.4)
    push_interval_s: tuple = (3.0, 6.0)
    push_vel: float = 0.25  # m/s horizontal velocity kick
    obs_noise: bool = True

    # floor debris + curriculum (level 0 = flat floor, 1 = hardest)
    debris: bool = True
    debris_max_height: float = 0.010  # protrusion at level 1 (m); swing target is 2.5 cm
    debris_ring: tuple = (0.15, 0.55)  # placed this far from the torso (m), never under the feet
    debris_level_init: float = 0.0
    debris_level_step: float = 0.05  # +step after a full-length episode, -step after a fall

    # rewards (each is multiplied by control_dt, like legged_gym / Isaac Lab)
    w_track_lin: float = 2.0
    w_track_ang: float = 1.0
    sigma_lin: float = 0.07  # m/s
    sigma_ang: float = 0.4  # rad/s
    w_alive: float = 1.0
    w_lin_vel_z: float = -2.0
    w_ang_vel_xy: float = -0.05
    w_orientation: float = -5.0
    w_base_height: float = -50.0
    w_torque: float = -0.01
    w_action_rate: float = -0.01
    w_joint_acc: float = -2.5e-7
    w_contact_phase: float = 0.6
    w_swing_height: float = -200.0
    swing_height: float = 0.025
    w_feet_slip: float = -1.0
    w_hip_pose: float = -1.0  # keep hip yaw/roll near default
    w_upper_pose: float = -0.3  # arms/head near default
    w_joint_limit: float = -2.0
    termination_penalty: float = -5.0


class PremaidWalkEnv:
    def __init__(self, cfg: WalkCfg | None = None, seed: int = 0, xml_path: str = ASSET_XML):
        self.cfg = cfg or WalkCfg()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.decimation = max(1, int(round(self.cfg.control_dt / m.opt.timestep)))
        self.dt = self.decimation * m.opt.timestep
        self.max_steps = int(round(self.cfg.episode_length_s / self.dt))

        # actuator / joint bookkeeping (actuator i drives joint SERVO_ORDER[i])
        names = [m.actuator(i).name for i in range(m.nu)]
        assert names == SERVO_ORDER, f"actuator order mismatch: {names}"
        jids = m.actuator_trnid[:, 0]
        self.qadr = m.jnt_qposadr[jids]
        self.vadr = m.jnt_dofadr[jids]
        self.default_q = np.array([DEFAULT_POSE.get(n, 0.0) for n in SERVO_ORDER])
        self.jnt_lo = m.jnt_range[jids, 0].copy()
        self.jnt_hi = m.jnt_range[jids, 1].copy()
        is_leg = np.array([n in LEG_JOINTS for n in SERVO_ORDER])
        self.action_scale = np.where(is_leg, self.cfg.action_scale_leg, self.cfg.action_scale_upper)
        self.hip_idx = np.array([SERVO_ORDER.index(n) for n in ("r_hpz", "l_hpz", "r_hpx", "l_hpx")])
        self.upper_idx = np.array([SERVO_ORDER.index(n) for n in UPPER_JOINTS])

        self.base_id = m.body("base_link").id
        self.floor_id = m.geom("floor").id
        self.sole_ids = np.array([m.geom("l_sole").id, m.geom("r_sole").id])  # order: left, right
        self.foot_site_ids = np.array([m.site("l_foot_site").id, m.site("r_foot_site").id])
        self.foot_geoms_body = {m.geom_bodyid[g] for g in self.sole_ids}
        self._sid = {n: m.sensor(n).adr[0] for n in ("gyro", "local_linvel")}
        self.debris_ids = np.array([i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("debris")], dtype=int)
        self.ground_ids = set([self.floor_id, *self.debris_ids.tolist()])
        self.debris_mocap = m.body_mocapid[m.geom_bodyid[self.debris_ids]]
        self.debris_half_h = m.geom_size[self.debris_ids, 2].copy()
        self.debris_level = self.cfg.debris_level_init

        # nominal model params for randomization
        self._nom_friction = m.geom_friction[:, 0].copy()
        self._nom_mass = m.body_mass[self.base_id]
        self._nom_kp = m.actuator_gainprm[:, 0].copy()
        self._nom_damping = m.dof_damping.copy()

        # standing height for the default pose (feet flat on the floor)
        self.stand_height = self._place_default(0.0, apply=False)

        self.commands = np.zeros(3)
        self.manual_commands = False  # set True to drive commands from outside (play/teleop)
        self.reset()

    # ------------------------------------------------------------------ utils
    def _place_default(self, yaw: float, apply: bool = True) -> float:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = self.default_q
        d.qpos[2] = 0.5
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_kinematics(m, d)
        # lowest point of the sole boxes: center z minus the box half-extent along world z
        lowest = min(
            d.geom_xpos[g][2] - float(np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g])
            for g in self.sole_ids
        )
        d.qpos[2] -= lowest
        mujoco.mj_forward(m, d)
        return float(d.qpos[2])

    def _randomize(self):
        c, m, r = self.cfg, self.model, self.rng
        if not c.randomize:
            return
        mu = r.uniform(*c.friction_range)
        m.geom_friction[self.floor_id, 0] = mu
        m.geom_friction[self.sole_ids, 0] = mu
        m.body_mass[self.base_id] = self._nom_mass * r.uniform(*c.body_mass_scale)
        kp = self._nom_kp * r.uniform(*c.kp_scale, size=m.nu)
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.dof_damping[:] = self._nom_damping * r.uniform(*c.damping_scale)
        self.friction = mu

    def _sample_commands(self):
        c, r = self.cfg, self.rng
        if r.random() < c.cmd_stand_prob:
            self.commands[:] = 0.0
        else:
            self.commands[:] = [r.uniform(*c.cmd_vx), r.uniform(*c.cmd_vy), r.uniform(*c.cmd_yaw)]
            self.commands[np.abs(self.commands) < c.cmd_deadband] = 0.0

    @property
    def walking(self) -> bool:
        return bool(np.any(np.abs(self.commands) > 0.0))

    def _gait_clock(self):
        if not self.walking:
            return np.zeros(2), np.array([True, True])
        ph = (self.step_count * self.dt / self.cfg.gait_period) % 1.0
        # left foot stance in first half of the cycle, right foot in the second half
        stance = np.array([ph < 0.5, ph >= 0.5])
        return np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]), stance

    def _foot_contacts(self):
        d = self.data
        contact = np.zeros(2, dtype=bool)
        bad = False
        if d.ncon:
            g = d.contact.geom[: d.ncon]
            for g1, g2 in g:
                g1_ground, g2_ground = g1 in self.ground_ids, g2 in self.ground_ids
                if g1_ground != g2_ground:
                    ground, other = (g1, g2) if g1_ground else (g2, g1)
                    if other == self.sole_ids[0]:
                        contact[0] = True
                    elif other == self.sole_ids[1]:
                        contact[1] = True
                    elif ground == self.floor_id:
                        bad = True  # torso/arm/shin on the floor = fallen (bumping debris is allowed)
        return contact, bad

    # ------------------------------------------------------------------ debris
    def _place_debris(self, idx, center_xy):
        """Drop debris pieces `idx` on a ring around center_xy, sticking up by a random amount."""
        c, d, r = self.cfg, self.data, self.rng
        n = len(idx)
        if n == 0:
            return
        mid = self.debris_mocap[idx]
        if not c.debris or self.debris_level <= 0.0:
            d.mocap_pos[mid] = [0.0, 0.0, -1.0]
            return
        ang = r.uniform(0, 2 * np.pi, n)
        rad = r.uniform(*c.debris_ring, n)
        prot = r.uniform(0.2, 1.0, n) * c.debris_max_height * self.debris_level
        d.mocap_pos[mid, 0] = center_xy[0] + rad * np.cos(ang)
        d.mocap_pos[mid, 1] = center_xy[1] + rad * np.sin(ang)
        d.mocap_pos[mid, 2] = prot - self.debris_half_h[idx]
        yaw = r.uniform(0, np.pi, n)
        d.mocap_quat[mid] = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], 1)

    def _recycle_debris(self):
        """Debris the robot has walked away from is moved back around it."""
        if not self.cfg.debris or self.debris_level <= 0.0 or len(self.debris_ids) == 0:
            return
        base = self.data.qpos[:2]
        dist = np.linalg.norm(self.data.mocap_pos[self.debris_mocap, :2] - base, axis=1)
        far = np.nonzero(dist > self.cfg.debris_ring[1] + 0.1)[0]
        if len(far):
            self._place_debris(far, base)

    def _update_curriculum(self, terminated_early: bool):
        c = self.cfg
        if not c.debris:
            return
        delta = -c.debris_level_step if terminated_early else c.debris_level_step
        self.debris_level = float(np.clip(self.debris_level + delta, 0.0, 1.0))

    # ------------------------------------------------------------------ API
    def reset(self):
        c, r = self.cfg, self.rng
        self._randomize()
        if not hasattr(self, "friction"):
            self.friction = float(self.model.geom_friction[self.floor_id, 0])
        self._place_default(r.uniform(-np.pi, np.pi))
        self._place_debris(np.arange(len(self.debris_ids)), (0.0, 0.0))
        d = self.data
        d.qpos[self.qadr] += r.normal(0, 0.03, size=NUM_ACTIONS)
        d.qvel[:6] += r.normal(0, 0.05, size=6)
        mujoco.mj_forward(self.model, d)
        d.ctrl[:] = self.default_q

        self.step_count = 0
        self.last_action = np.zeros(NUM_ACTIONS)
        self.last_dq = np.zeros(NUM_ACTIONS)
        self.foot_pos_prev = d.site_xpos[self.foot_site_ids].copy()
        self.next_push = r.uniform(*c.push_interval_s)
        self.push_flag = 0.0
        if not self.manual_commands:
            self._sample_commands()
        self.ep_sums = {}
        obs, state = self._observe()
        return obs, state

    def _observe(self):
        c, d, r = self.cfg, self.data, self.rng
        R = d.xmat[self.base_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[self._sid["gyro"]: self._sid["gyro"] + 3].copy()
        linvel = d.sensordata[self._sid["local_linvel"]: self._sid["local_linvel"] + 3].copy()
        q = d.qpos[self.qadr] - self.default_q
        dq = d.qvel[self.vadr].copy()
        clock, _ = self._gait_clock()
        cmd_scaled = self.commands * np.array([4.0, 4.0, 1.0])

        clean = np.concatenate([gyro * 0.25, grav, cmd_scaled, q, dq * 0.05, self.last_action, clock])
        if c.obs_noise:
            noise = np.concatenate([
                r.normal(0, 0.1, 3) * 0.25, r.normal(0, 0.03, 3), np.zeros(3),
                r.normal(0, 0.01, 25), r.normal(0, 0.5, 25) * 0.05, np.zeros(25), np.zeros(2),
            ])
            obs = clean + noise
        else:
            obs = clean
        contact, _ = self._foot_contacts()
        foot_z = d.site_xpos[self.foot_site_ids, 2]
        state = np.concatenate([
            clean, linvel * 2.0, [d.qpos[2] - self.stand_height], contact.astype(float), foot_z * 10.0,
            [self.friction], [self.push_flag],
        ])
        return obs.astype(np.float32), state.astype(np.float32)

    def step(self, action: np.ndarray):
        c, m, d, r = self.cfg, self.model, self.data, self.rng
        action = np.clip(np.asarray(action, dtype=np.float64), -c.action_clip, c.action_clip)
        d.ctrl[:] = self.default_q + self.action_scale * action

        # random push: kick the base velocity
        t = self.step_count * self.dt
        self.push_flag = 0.0
        if c.randomize and t >= self.next_push:
            ang = r.uniform(0, 2 * np.pi)
            d.qvel[0:2] += c.push_vel * np.array([np.cos(ang), np.sin(ang)])
            self.next_push = t + r.uniform(*c.push_interval_s)
            self.push_flag = 1.0

        torque_sq = 0.0
        self._recycle_debris()
        for _ in range(self.decimation):
            mujoco.mj_step(m, d)
            torque_sq += float(np.sum(np.square(d.actuator_force)))
        torque_sq /= self.decimation
        self.step_count += 1

        if not self.manual_commands and self.step_count % int(round(c.cmd_resample_s / self.dt)) == 0:
            self._sample_commands()

        rew, terms, terminated = self._reward(action, torque_sq)
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            self._update_curriculum(terminated_early=terminated)
        for k, v in terms.items():
            self.ep_sums[k] = self.ep_sums.get(k, 0.0) + v

        self.last_action = action
        obs, state = self._observe()
        return obs, state, rew, terminated, truncated

    def _reward(self, action, torque_sq):
        c, d, dt = self.cfg, self.data, self.dt
        R = d.xmat[self.base_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[self._sid["gyro"]: self._sid["gyro"] + 3]
        linvel = d.sensordata[self._sid["local_linvel"]: self._sid["local_linvel"] + 3]
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        contact, bad_contact = self._foot_contacts()
        _, stance = self._gait_clock()
        foot_pos = d.site_xpos[self.foot_site_ids]
        foot_vel = (foot_pos - self.foot_pos_prev) / dt
        self.foot_pos_prev = foot_pos.copy()

        terms = {}
        lin_err = np.sum(np.square(self.commands[:2] - linvel[:2]))
        terms["track_lin"] = c.w_track_lin * np.exp(-lin_err / c.sigma_lin**2)
        ang_err = (self.commands[2] - gyro[2]) ** 2
        terms["track_ang"] = c.w_track_ang * np.exp(-ang_err / c.sigma_ang**2)
        terms["alive"] = c.w_alive
        terms["lin_vel_z"] = c.w_lin_vel_z * linvel[2] ** 2
        terms["ang_vel_xy"] = c.w_ang_vel_xy * np.sum(np.square(gyro[:2]))
        terms["orientation"] = c.w_orientation * np.sum(np.square(grav[:2]))
        terms["base_height"] = c.w_base_height * (d.qpos[2] - self.stand_height) ** 2
        terms["torque"] = c.w_torque * torque_sq
        terms["action_rate"] = c.w_action_rate * np.sum(np.square(action - self.last_action))
        terms["joint_acc"] = c.w_joint_acc * np.sum(np.square((dq - self.last_dq) / dt))
        self.last_dq = dq.copy()
        terms["contact_phase"] = c.w_contact_phase * np.mean(contact == stance)
        if self.walking:
            swing = ~stance
            err = np.square(foot_pos[:, 2] - c.swing_height) * swing
            terms["swing_height"] = c.w_swing_height * np.sum(err)
        else:
            terms["swing_height"] = 0.0
        terms["feet_slip"] = c.w_feet_slip * np.sum(np.sum(np.square(foot_vel[:, :2]), axis=1) * contact)
        terms["hip_pose"] = c.w_hip_pose * np.sum(np.square(q[self.hip_idx] - self.default_q[self.hip_idx]))
        terms["upper_pose"] = c.w_upper_pose * np.sum(np.square(q[self.upper_idx] - self.default_q[self.upper_idx]))
        span = self.jnt_hi - self.jnt_lo
        soft_lo, soft_hi = self.jnt_lo + 0.05 * span, self.jnt_hi - 0.05 * span
        terms["joint_limit"] = c.w_joint_limit * np.sum(np.clip(soft_lo - q, 0, None) + np.clip(q - soft_hi, 0, None))

        terms = {k: float(v) * dt for k, v in terms.items()}

        fallen = d.qpos[2] < c.min_base_height or np.linalg.norm(grav[:2]) > np.sqrt(1 - c.max_tilt_cos**2) or bad_contact
        terminated = bool(fallen)
        terms["termination"] = c.termination_penalty if terminated else 0.0
        return float(sum(terms.values())), terms, terminated

    def episode_summary(self):
        """Per-second averages of each reward term over the finished episode (for logging)."""
        T = max(self.step_count * self.dt, 1e-6)
        out = {f"Episode_Reward/{k}": v / T * (1.0 if k != "termination" else T) for k, v in self.ep_sums.items()}
        out["Episode/length_s"] = self.step_count * self.dt
        out["Curriculum/debris_level"] = self.debris_level
        return out
