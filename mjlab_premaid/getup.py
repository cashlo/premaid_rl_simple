"""Premaid AI get-up task for mjlab (MuJoCo Warp, GPU).

Port of mujoco_sim/getup_env.py. Same task, rewards, hand-over success test, action filter and
86-value actor observation (so mujoco_sim/deploy/deploy_policy.py can run the result), now
batched on the GPU:

  * start poses: lying (face up/down/side), squat, sitting, kneeling/all-fours pools, generated
    once on the CPU with the MuJoCo version and replayed here with a random heading
  * servo model per physics substep: command delay 0-20 ms, slew limit ~220 deg/s, per-servo
    settle offset, joint readings 0-20 ms stale (all randomized per episode)
  * success = hand-over pose for the walking policy (upright, legs near the crouch, smoothed gyro
    low, hips up) held 0.2 s -> episode ends with the remaining standing reward as a bonus
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .robot import DEFAULT_POSE, JOINT_NAMES, LEG_JOINTS, SERVO_ORDER, premaid_robot_cfg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mujoco_sim"))

NUM_ACTOR_OBS = 86
START_KINDS = ("lying", "squat", "sit", "kneel")


@dataclass
class GetUpParams:
    """Everything that isn't plain mjlab config (mirrors mujoco_sim/getup_env.GetUpCfg)."""

    episode_length_s: float = 10.0
    # start mix
    start_probs: tuple = (0.45, 0.25, 0.15, 0.15)  # lying, squat, sit, kneel
    pool_sizes: tuple = (256, 96, 96, 96)
    init_noise: float = 0.05
    # rewards
    w_upright: float = 2.0
    w_height: float = 2.0
    w_hip_height: float = 2.0
    w_feet_down: float = 1.0
    w_leg_pose: float = 1.0
    leg_pose_sigma: float = 2.0
    w_stand_pose: float = 2.0
    w_stand_feet: float = 1.0
    w_stand_still: float = 1.0
    w_action_rate: float = -0.005
    w_torque: float = -0.01
    w_joint_vel: float = -0.0005
    w_ang_vel: float = -0.02
    w_saturation: float = -0.2
    w_joint_limit: float = -2.0
    # success / hand-over
    success_upright: float = 0.9
    success_leg_err: float = 0.8
    success_gyro: float = 0.6
    success_gyro_alpha: float = 0.15
    success_hip_frac: float = 0.8
    success_hold_s: float = 0.2
    success_rate: float = 12.0
    # robot geometry (from the CPU model's default crouch)
    stand_height: float = 0.2416
    stand_head_z: float = 0.398


P = GetUpParams()


# ======================================================================================
# Servo model + action
# ======================================================================================
@dataclass(kw_only=True)
class PremaidServoActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    scale: float = 1.0  # rad per action unit (get-up: 1.0 everywhere)
    action_clip: float = 3.0
    filter_alpha: float = 0.3
    vmax: float = 3.9
    vmax_scale: tuple = (0.8, 1.2)
    cmd_delay_steps: tuple = (0, 4)
    obs_delay_steps: tuple = (0, 4)
    servo_offset: float = 0.02
    randomize: bool = True

    def build(self, env) -> "PremaidServoAction":
        return PremaidServoAction(self, env)


class PremaidServoAction(ActionTerm):
    """Policy action (servo order) -> filtered target -> delay -> slew limit -> position servo."""

    cfg: PremaidServoActionCfg

    def __init__(self, cfg: PremaidServoActionCfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        ids, _ = self._entity.find_joints(JOINT_NAMES, preserve_order=True)
        self.joint_ids = torch.tensor(ids, device=dev, dtype=torch.long)
        self.default_q = self._entity.data.default_joint_pos[:, self.joint_ids].clone()
        lim = self._entity.data.joint_pos_limits[:, self.joint_ids]
        self.lo, self.hi = lim[..., 0].clone(), lim[..., 1].clone()
        self.L = cfg.cmd_delay_steps[1] + 1
        self.H = cfg.obs_delay_steps[1] + 1
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.filt, self.last, self.prev = z(n, 25), z(n, 25), z(n, 25)
        self.target, self.ref = z(n, 25), z(n, 25)
        self.cmd_buf = z(n, self.L, 25)
        self.q_hist, self.dq_hist = z(n, self.H, 25), z(n, self.H, 25)
        self.cmd_idx = 0
        self.hist_idx = 0
        self.cmd_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.obs_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.vmax = torch.full((n, 1), cfg.vmax, device=dev)
        self.bias = z(n, 25)
        self.needs_init = torch.ones(n, dtype=torch.bool, device=dev)
        self.ar = torch.arange(n, device=dev)
        self.physics_dt = env.physics_dt

    @property
    def action_dim(self) -> int:
        return 25

    @property
    def raw_action(self) -> torch.Tensor:
        return self.last

    def _current(self):
        d = self._entity.data
        return d.joint_pos[:, self.joint_ids], d.joint_vel[:, self.joint_ids]

    def _init_envs(self):
        ids = self.needs_init.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        q, dq = self._current()
        q, dq = q[ids], dq[ids]
        a0 = ((q - self.default_q[ids]) / self.cfg.scale).clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self.filt[ids] = a0
        self.last[ids] = a0
        self.prev[ids] = a0
        self.ref[ids] = q
        self.target[ids] = q
        self.cmd_buf[ids] = q.unsqueeze(1)
        self.q_hist[ids] = q.unsqueeze(1)
        self.dq_hist[ids] = dq.unsqueeze(1) * 0.0
        self.needs_init[ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        self._init_envs()
        a = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        al = self.cfg.filter_alpha
        self.prev = self.last.clone()
        self.filt = al * a + (1.0 - al) * self.filt
        self.last = self.filt.clone()
        self.target = torch.max(torch.min(self.default_q + self.cfg.scale * self.filt + self.bias, self.hi), self.lo)

    def apply_actions(self) -> None:
        self.cmd_buf[:, self.cmd_idx] = self.target
        delayed = self.cmd_buf[self.ar, (self.cmd_idx - self.cmd_lag) % self.L]
        self.cmd_idx = (self.cmd_idx + 1) % self.L
        step = self.vmax * self.physics_dt
        self.ref = self.ref + torch.max(torch.min(delayed - self.ref, step), -step)
        self._entity.set_joint_position_target(self.ref, joint_ids=self.joint_ids)
        q, dq = self._current()
        self.q_hist[:, self.hist_idx] = q
        self.dq_hist[:, self.hist_idx] = dq
        self.hist_idx = (self.hist_idx + 1) % self.H

    def reading(self):
        """Joint position/velocity as the RCB-4 would report them (stale by obs_lag substeps)."""
        q_now, dq_now = self._current()
        k = (self.hist_idx - 1 - self.obs_lag) % self.H
        q, dq = self.q_hist[self.ar, k], self.dq_hist[self.ar, k]
        init = self.needs_init.unsqueeze(1)
        return torch.where(init, q_now, q), torch.where(init, dq_now, dq)

    def obs_last_action(self):
        q_now, _ = self._current()
        a0 = ((q_now - self.default_q) / self.cfg.scale).clamp(-self.cfg.action_clip, self.cfg.action_clip)
        return torch.where(self.needs_init.unsqueeze(1), a0, self.last)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self.ar
        c, k = self.cfg, len(env_ids)
        dev = self.device
        if c.randomize:
            self.vmax[env_ids, 0] = c.vmax * torch.empty(k, device=dev).uniform_(*c.vmax_scale)
            self.cmd_lag[env_ids] = torch.randint(c.cmd_delay_steps[0], c.cmd_delay_steps[1] + 1, (k,), device=dev)
            self.obs_lag[env_ids] = torch.randint(c.obs_delay_steps[0], c.obs_delay_steps[1] + 1, (k,), device=dev)
            self.bias[env_ids] = torch.empty(k, 25, device=dev).uniform_(-c.servo_offset, c.servo_offset)
        else:
            self.vmax[env_ids, 0] = c.vmax
            self.cmd_lag[env_ids] = (c.cmd_delay_steps[0] + c.cmd_delay_steps[1]) // 2
            self.obs_lag[env_ids] = (c.obs_delay_steps[0] + c.obs_delay_steps[1]) // 2
            self.bias[env_ids] = 0.0
        self.needs_init[env_ids] = True
        return {}


def _act(env) -> PremaidServoAction:
    return env.action_manager.get_term("servo")


# ======================================================================================
# Per-env task state (start kind, hand-over counters)
# ======================================================================================
class _State:
    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.kind = torch.zeros(n, dtype=torch.long, device=dev)
        self.gyro_ema = torch.zeros(n, 3, device=dev)
        self.hold = torch.zeros(n, dtype=torch.long, device=dev)
        robot = env.scene["robot"]
        self.head_id = robot.find_bodies("head")[0][0]
        self.leg_idx = torch.tensor([SERVO_ORDER.index(n) for n in LEG_JOINTS], device=dev)
        self.pools = None


def _state(env) -> _State:
    if not hasattr(env, "_premaid_getup"):
        env._premaid_getup = _State(env)
    return env._premaid_getup


def _upright_head_hip(env):
    d = env.scene["robot"].data
    upright = (-d.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    hip = (d.root_link_pos_w[:, 2] / P.stand_height).clamp(0.0, 1.1)
    head = (d.body_link_pos_w[:, _state(env).head_id, 2] / P.stand_head_z).clamp(0.0, 1.1)
    return upright, head, hip


def _feet(env):
    return env.scene["feet_contact"].data.found > 0  # (N, 2) left, right


def _standing(env):
    upright, _, hip = _upright_head_hip(env)
    return (upright > 0.9) & (hip > 0.85)


# ======================================================================================
# Observations (same 86-value layout as mujoco_sim/premaid_env.py)
# ======================================================================================
def actor_obs(env, noise: bool = True) -> torch.Tensor:
    d = env.scene["robot"].data
    act = _act(env)
    q, dq = act.reading()
    gyro = d.root_link_ang_vel_b
    grav = d.projected_gravity_b
    n = env.num_envs
    zeros3, zeros2 = torch.zeros(n, 3, device=env.device), torch.zeros(n, 2, device=env.device)
    qrel = q - act.default_q
    if noise:
        gyro = gyro + torch.randn_like(gyro) * 0.1
        grav = grav + torch.randn_like(grav) * 0.03
        qrel = qrel + torch.randn_like(qrel) * 0.01
        dq = dq + torch.randn_like(dq) * 0.5
    return torch.cat([gyro * 0.25, grav, zeros3, qrel, dq * 0.05, act.obs_last_action(), zeros2], dim=-1)


def critic_obs(env) -> torch.Tensor:
    d = env.scene["robot"].data
    feet = _feet(env).float()
    upright, head, hip = _upright_head_hip(env)
    st = _state(env)
    return torch.cat([
        actor_obs(env, noise=False),
        d.root_link_lin_vel_b * 2.0,
        (d.root_link_pos_w[:, 2:3] - P.stand_height),
        feet,
        torch.stack([upright, head, hip], dim=-1),
        st.gyro_ema,
        (st.hold.float() * env.step_dt).unsqueeze(-1),
    ], dim=-1)


# ======================================================================================
# Rewards (functions return per-second values; mjlab multiplies by weight * dt)
# ======================================================================================
def r_upright(env):
    return _upright_head_hip(env)[0]


def r_head(env):
    return _upright_head_hip(env)[1].clamp(max=1.0)


def r_hip(env):
    up, _, hip = _upright_head_hip(env)
    return hip.clamp(max=1.0) * up


def r_feet_down(env):
    return _feet(env).all(dim=-1).float() * _upright_head_hip(env)[0]


def r_leg_pose(env):
    act, st = _act(env), _state(env)
    q, _ = act._current()
    err = ((q - act.default_q)[:, st.leg_idx] ** 2).sum(-1)
    return torch.exp(-err / P.leg_pose_sigma) * _upright_head_hip(env)[0]


def r_stand_pose(env):
    act = _act(env)
    q, _ = act._current()
    return torch.exp(-((q - act.default_q) ** 2).sum(-1) / 0.5) * _standing(env).float()


def r_stand_feet(env):
    return _feet(env).all(dim=-1).float() * _standing(env).float()


def r_stand_still(env):
    d = env.scene["robot"].data
    v = (d.root_link_lin_vel_w ** 2).sum(-1) + (d.root_link_ang_vel_w ** 2).sum(-1)
    return torch.exp(-v / 0.1) * _standing(env).float()


def r_action_rate(env):
    act = _act(env)
    return ((act.last - act.prev) ** 2).sum(-1)


def r_torque(env):
    return (env.scene["robot"].data.actuator_force ** 2).sum(-1)


def r_joint_vel(env):
    return (_act(env)._current()[1] ** 2).sum(-1)


def r_ang_vel(env):
    return (env.scene["robot"].data.root_link_ang_vel_b ** 2).sum(-1)


def r_saturation(env):
    f = env.scene["robot"].data.actuator_force.abs()
    lim = _effort_limits(env)
    return (f > 0.95 * lim).float().mean(-1)


def r_joint_limit(env):
    act = _act(env)
    q, _ = act._current()
    span = act.hi - act.lo
    lo, hi = act.lo + 0.05 * span, act.hi - 0.05 * span
    return ((lo - q).clamp(min=0) + (q - hi).clamp(min=0)).sum(-1)


def r_success(env):
    ok = env.termination_manager.get_term("success").float()
    remaining = (P.episode_length_s - env.episode_length_buf.float() * env.step_dt).clamp(min=0.0)
    return ok * P.success_rate * remaining / env.step_dt  # manager multiplies by dt


def _effort_limits(env):
    if not hasattr(env, "_premaid_effort"):
        robot = env.scene["robot"]
        lims = []
        for name in robot.actuator_names if hasattr(robot, "actuator_names") else []:
            lims.append(1.3712 if any(t in name for t in ("hp", "kny", "ak")) else 0.9114)
        if not lims:  # fall back to every actuator being a leg servo
            lims = [1.3712] * robot.data.actuator_force.shape[-1]
        env._premaid_effort = torch.tensor(lims, device=env.device)
    return env._premaid_effort


# ======================================================================================
# Terminations
# ======================================================================================
def t_success(env):
    st, act = _state(env), _act(env)
    d = env.scene["robot"].data
    st.gyro_ema = P.success_gyro_alpha * d.root_link_ang_vel_b + (1 - P.success_gyro_alpha) * st.gyro_ema
    upright, _, hip = _upright_head_hip(env)
    q, _ = act._current()
    leg_err = torch.linalg.norm((q - act.default_q)[:, st.leg_idx], dim=-1)
    ok = (upright > P.success_upright) & (leg_err < P.success_leg_err) & \
         (torch.linalg.norm(st.gyro_ema, dim=-1) < P.success_gyro) & (hip > P.success_hip_frac)
    st.hold = torch.where(ok, st.hold + 1, torch.zeros_like(st.hold))
    return st.hold * env.step_dt >= P.success_hold_s - 1e-6


# ======================================================================================
# Events: start poses + logging
# ======================================================================================
POOL_FILE = os.path.join(HERE, "assets", "getup_pools.npz")


def make_pool_file(path: str = POOL_FILE):
    """Generate start poses with the CPU MuJoCo version (slow-ish, done once) and save them by joint name."""
    from getup_env import GetUpCfg, PremaidGetUpEnv

    n_l, n_sq, n_si, n_kn = P.pool_sizes
    cpu = PremaidGetUpEnv(GetUpCfg(randomize=False, pool_size=n_l, squat_pool_size=n_sq,
                                   mid_pool_size=max(n_si, n_kn)), seed=0)
    m = cpu.model
    names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] != 0]  # hinge joints
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    out = {"joint_names": np.array(names)}
    for kind, pool in zip(START_KINDS, (cpu._pool, cpu._squat_pool, cpu._sit_pool, cpu._kneel_pool)):
        q = np.array([p[0] for p in pool])
        out[f"{kind}_root"] = q[:, :7].astype(np.float32)
        out[f"{kind}_joints"] = q[:, adr].astype(np.float32)
    np.savez_compressed(path, **out)
    print(f"saved {path}: " + ", ".join(f"{k} {len(out[k + '_root'])}" for k in START_KINDS))


def _build_pools(env):
    if not os.path.exists(POOL_FILE):
        make_pool_file()
    z = np.load(POOL_FILE)
    names = list(z["joint_names"])
    robot = env.scene["robot"]
    col = [names.index(n) for n in robot.joint_names]
    pools = []
    for kind in START_KINDS:
        root = z[f"{kind}_root"].copy()
        root[:, 2] += 0.005
        pools.append((torch.tensor(root, device=env.device), torch.tensor(z[f"{kind}_joints"][:, col], device=env.device)))
    return pools


def reset_start_pose(env, env_ids):
    st = _state(env)
    if st.pools is None:
        st.pools = _build_pools(env)
    robot = env.scene["robot"]
    k = len(env_ids)
    dev = env.device
    probs = torch.tensor(P.start_probs, device=dev)
    kinds = torch.multinomial(probs, k, replacement=True)
    st.kind[env_ids] = kinds
    root = torch.zeros(k, 7, device=dev)
    joints = torch.zeros(k, robot.num_joints, device=dev)
    for kk, (proot, pjoint) in enumerate(st.pools):
        sel = (kinds == kk).nonzero(as_tuple=False).squeeze(-1)
        if len(sel) == 0:
            continue
        pick = torch.randint(0, proot.shape[0], (len(sel),), device=dev)
        root[sel] = proot[pick]
        joints[sel] = pjoint[pick]
    # random heading: q = q_yaw * q
    yaw = torch.rand(k, device=dev) * 2 * math.pi - math.pi
    w1, z1 = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w2, x2, y2, z2 = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    root[:, 3:7] = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2], dim=-1)
    root[:, :2] = 0.0
    root[:, :3] += env.scene.env_origins[env_ids]
    lim = robot.data.joint_pos_limits[env_ids]
    joints = (joints + torch.randn_like(joints) * P.init_noise).clamp(lim[..., 0], lim[..., 1])
    robot.write_root_link_pose_to_sim(root, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(k, 6, device=dev), env_ids=env_ids)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
    st.gyro_ema[env_ids] = 0.0
    st.hold[env_ids] = 0


def log_success(env, env_ids):
    """Before the reset: success rate per start kind for the episodes that just ended."""
    if env.common_step_counter == 0:
        return
    st = _state(env)
    ok = env.termination_manager.get_term("success")[env_ids].float()
    t = env.episode_length_buf[env_ids].float() * env.step_dt
    for kk, name in enumerate(START_KINDS):
        sel = st.kind[env_ids] == kk
        if sel.any():
            env.extras["log"][f"GetUp/success_from_{name}"] = ok[sel].mean().item()
            done = sel & (ok > 0)
            if done.any():
                env.extras["log"][f"GetUp/time_to_success_s_from_{name}"] = t[done].mean().item()


# ======================================================================================
# Config
# ======================================================================================
def premaid_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet = ContactSensorCfg(
        name="feet_contact",
        primary=ContactMatch(mode="geom", pattern=("l_sole", "r_sole"), entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    W = lambda f, w: RewardTermCfg(func=f, weight=w)  # noqa: E731
    cfg = ManagerBasedRlEnvCfg(
        decimation=4,
        episode_length_s=P.episode_length_s,
        sim=SimulationCfg(
            nconmax=96,
            njmax=400,
            mujoco=MujocoCfg(timestep=0.005, integrator="implicitfast", cone="elliptic", impratio=10.0,
                             iterations=10, ls_iterations=20),
        ),
        scene=SceneCfg(
            num_envs=4096,
            env_spacing=1.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": premaid_robot_cfg()},
            sensors=(feet,),
        ),
        observations={
            "actor": ObservationGroupCfg(terms={"obs": ObservationTermCfg(func=actor_obs, params={"noise": not play})},
                                         concatenate_terms=True, enable_corruption=False),
            "critic": ObservationGroupCfg(terms={"obs": ObservationTermCfg(func=critic_obs)},
                                          concatenate_terms=True, enable_corruption=False),
        },
        actions={"servo": PremaidServoActionCfg(randomize=not play)},
        events={
            "floor_friction": EventTermCfg(
                mode="startup", func=dr.geom_friction,
                params={"asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)), "operation": "abs",
                        "axes": [0], "ranges": (0.4, 1.2)},
            ),
            "torso_mass": EventTermCfg(
                mode="startup", func=dr.body_mass,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)), "operation": "scale",
                        "ranges": (0.9, 1.15)},
            ),
            "servo_gains": EventTermCfg(
                mode="reset", func=dr.pd_gains,
                params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.8, 1.2), "kd_range": (0.7, 1.4),
                        "operation": "scale"},
            ),
            "log_success": EventTermCfg(mode="reset", func=log_success),
            "start_pose": EventTermCfg(mode="reset", func=reset_start_pose),
        },
        rewards={
            "upright": W(r_upright, P.w_upright),
            "head_height": W(r_head, P.w_height),
            "hip_height": W(r_hip, P.w_hip_height),
            "feet_down": W(r_feet_down, P.w_feet_down),
            "leg_pose": W(r_leg_pose, P.w_leg_pose),
            "stand_pose": W(r_stand_pose, P.w_stand_pose),
            "stand_feet": W(r_stand_feet, P.w_stand_feet),
            "stand_still": W(r_stand_still, P.w_stand_still),
            "action_rate": W(r_action_rate, P.w_action_rate),
            "torque": W(r_torque, P.w_torque),
            "joint_vel": W(r_joint_vel, P.w_joint_vel),
            "ang_vel": W(r_ang_vel, P.w_ang_vel),
            "saturation": W(r_saturation, P.w_saturation),
            "joint_limit": W(r_joint_limit, P.w_joint_limit),
            "success": W(r_success, 1.0),
        },
        terminations={
            "success": TerminationTermCfg(func=t_success),
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        viewer=ViewerConfig(body_name="base_link", distance=1.0, elevation=-15.0),
    )
    if play:
        cfg.episode_length_s = 20.0
        cfg.events.pop("servo_gains")
    return cfg
