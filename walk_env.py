"""Premaid AI velocity-command walking task for Isaac Lab (DirectRLEnv).

Same task design as the MuJoCo version in mujoco_sim/premaid_env.py:
  * small, per-joint action scale around a crouched default pose
  * gait clock + foot-contact-phase / swing-height rewards
  * commands sized for a 0.38 m robot, resampled per robot on its own timer
  * rewards scaled by dt, modest termination penalty
  * real velocity-kick pushes per robot (not a persistent external force)
  * friction / mass / servo-gain randomization
  * actor sees only what the real robot can sense; the critic also gets
    privileged terms (packed at the end of the observation, see NUM_ACTOR_OBS)

Used by train_walk.py and play_walk.py. Your original train_direct.py / env_cfg.py are untouched.
"""

from __future__ import annotations

import math

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

# ==============================================================================
# Robot
# ==============================================================================
DEFAULT_POSE = {
    "r_hpy_joint": -0.35, "l_hpy_joint": -0.35,
    "r_kny_joint": 0.70, "l_kny_joint": 0.70,
    "r_aky_joint": -0.35, "l_aky_joint": -0.35,
    "r_shx_joint": 1.35, "l_shx_joint": -1.35,
}  # every other joint starts at 0
LEG_JOINTS = ".*_(hp[xyz]|kny|ak[xy])_joint"
UPPER_JOINTS = ".*_(sh[xy]|el[xy]|wry)_joint|head_.*"

PREMAID_WALK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # premaidai_fixed.usda = your premaidai.usd with link inertias corrected (they were 100x too large)
        usd_path="premaidai_fixed.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # arms hang right next to the torso hulls; leave self-collision off (the hip-pose reward keeps legs apart)
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30), joint_pos=DEFAULT_POSE),
    # Same servo model as the MuJoCo version (guesses until fitted to the rcb4 step-response data).
    # URDF torque limits: 1.37 N*m legs, 0.91 N*m arms/head.
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[LEG_JOINTS], stiffness=6.0, damping=0.08, armature=0.002,
            effort_limit_sim=1.37, velocity_limit_sim=7.5,
        ),
        "upper": ImplicitActuatorCfg(
            joint_names_expr=[UPPER_JOINTS], stiffness=3.0, damping=0.08, armature=0.002,
            effort_limit_sim=0.91, velocity_limit_sim=5.8,
        ),
    },
)


# ==============================================================================
# Domain randomization (Isaac Lab event manager)
# ==============================================================================
@configclass
class WalkEventCfg:
    friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    torso_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link|body"),
            "mass_distribution_params": (0.9, 1.15),
            "operation": "scale",
        },
    )
    servo_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.7, 1.4),
            "operation": "scale",
        },
    )


# ==============================================================================
# Task config
# ==============================================================================
NUM_ACTOR_OBS = 3 + 3 + 3 + 25 + 25 + 25 + 2  # = 86, same layout as the MuJoCo actor
NUM_PRIV_OBS = 3 + 1 + 2 + 2 + 1  # lin vel, height error, foot contacts, foot heights, push flag


@configclass
class PremaidWalkEnvCfg(DirectRLEnvCfg):
    # 200 Hz physics, 50 Hz policy
    decimation = 4
    episode_length_s = 20.0
    action_space = 25
    observation_space = NUM_ACTOR_OBS + NUM_PRIV_OBS
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=512 * 1024,
            gpu_max_rigid_contact_count=2048 * 1024,
            gpu_found_lost_pairs_capacity=512 * 1024,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)
    robot: ArticulationCfg = PREMAID_WALK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.0
    )
    events: WalkEventCfg = WalkEventCfg()

    foot_body_names = ("l_foot", "r_foot")  # order: left, right
    sole_offset = 0.018  # foot body origin -> bottom of the sole (m), from the foot mesh

    # actions
    action_scale_leg = 0.30
    action_scale_upper = 0.20
    action_clip = 3.0

    # commands (vx, vy m/s, yaw rad/s)
    cmd_vx = (-0.10, 0.20)
    cmd_vy = (-0.08, 0.08)
    cmd_yaw = (-0.6, 0.6)
    cmd_resample_s = 5.0
    cmd_stand_prob = 0.15
    cmd_deadband = 0.03
    gait_period = 0.5

    # pushes: horizontal velocity kick every 3-6 s per robot
    push_interval_s = (3.0, 6.0)
    push_vel = 0.25

    # termination
    min_height_frac = 0.62  # of standing height
    max_tilt = 0.8  # |gravity_xy| in body frame (~53 deg)

    # rewards (multiplied by step dt)
    w_track_lin = 2.0
    w_track_ang = 1.0
    sigma_lin = 0.07
    sigma_ang = 0.4
    w_alive = 1.0
    w_lin_vel_z = -2.0
    w_ang_vel_xy = -0.05
    w_orientation = -5.0
    w_base_height = -50.0
    w_torque = -0.01
    w_action_rate = -0.01
    w_joint_acc = -2.5e-7
    w_contact_phase = 0.6
    w_swing_height = -200.0
    swing_height = 0.025
    w_feet_slip = -1.0
    w_hip_pose = -1.0
    w_upper_pose = -0.3
    w_joint_limit = -2.0
    termination_penalty = -5.0

    obs_noise = True


# ==============================================================================
# Environment
# ==============================================================================
class PremaidWalkEnv(DirectRLEnv):
    cfg: PremaidWalkEnvCfg

    def __init__(self, cfg: PremaidWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        r = self.robot
        n, dev = self.num_envs, self.device

        self.default_q = r.data.default_joint_pos.clone()
        leg_ids, _ = r.find_joints(LEG_JOINTS)
        self.action_scale = torch.full((r.num_joints,), cfg.action_scale_upper, device=dev)
        self.action_scale[leg_ids] = cfg.action_scale_leg
        self.hip_ids, _ = r.find_joints(["r_hpz_joint", "l_hpz_joint", "r_hpx_joint", "l_hpx_joint"])
        self.upper_ids, _ = r.find_joints(UPPER_JOINTS)
        assert r.num_joints == 25, f"expected 25 joints, got {r.num_joints}: {r.joint_names}"

        try:
            self.foot_ids, _ = r.find_bodies(list(cfg.foot_body_names), preserve_order=True)
            self.foot_sensor_ids, _ = self.contact_sensor.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        except ValueError as e:
            raise ValueError(f"{e}\nRobot bodies are: {r.body_names}\nSet PremaidWalkEnvCfg.foot_body_names.") from e

        # buffers
        self.actions = torch.zeros(n, 25, device=dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dq = torch.zeros(n, r.num_joints, device=dev)
        self.commands = torch.zeros(n, 3, device=dev)
        self.cmd_timer = torch.zeros(n, device=dev)
        self.push_timer = torch.zeros(n, device=dev)
        self.push_flag = torch.zeros(n, device=dev)
        self.manual_commands = False  # play_walk.py drives commands from the keyboard
        self.episode_sums: dict[str, torch.Tensor] = {}

        self._targets = self.default_q.clone()

        # standing height from the default pose (works whatever the USD's root frame is):
        # put every robot in the default pose, take one tiny physics step, measure root-to-sole height.
        r.write_joint_state_to_sim(self.default_q, torch.zeros_like(self.default_q))
        r.set_joint_position_target(self.default_q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.physics_dt)
        root_z = r.data.root_pos_w[:, 2]
        foot_z = r.data.body_pos_w[:, self.foot_ids, 2].mean(dim=1)
        self.stand_height = float((root_z - foot_z).mean()) + cfg.sole_offset
        print(f"[walk_env] standing root height {self.stand_height:.3f} m, joints: {r.joint_names}")

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        ground = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8)
        )
        ground.func("/World/defaultGroundPlane", ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ helpers
    def _sample_commands(self, ids: torch.Tensor):
        c, k = self.cfg, len(ids)
        if k == 0 or self.manual_commands:
            return
        u = lambda lo_hi: torch.empty(k, device=self.device).uniform_(*lo_hi)  # noqa: E731
        cmd = torch.stack([u(c.cmd_vx), u(c.cmd_vy), u(c.cmd_yaw)], dim=1)
        cmd[cmd.abs() < c.cmd_deadband] = 0.0
        stand = torch.rand(k, device=self.device) < c.cmd_stand_prob
        cmd[stand] = 0.0
        self.commands[ids] = cmd
        self.cmd_timer[ids] = 0.0

    def _gait(self):
        walking = (self.commands.abs() > 0.0).any(dim=1)
        t = self.episode_length_buf.float() * self.step_dt
        phase = (t / self.cfg.gait_period) % 1.0
        clock = torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1)
        clock = clock * walking.unsqueeze(1)
        stance = torch.stack([phase < 0.5, phase >= 0.5], dim=1)  # left, right
        stance[~walking] = True
        return clock, stance, walking

    def _foot_contacts(self):
        f = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_sensor_ids, :]
        return f.norm(dim=-1).max(dim=1).values > 1.0  # (n, 2)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        self.actions = actions.clamp(-c.action_clip, c.action_clip)
        self._targets = self.default_q + self.action_scale * self.actions

        # per-robot command resampling
        self.cmd_timer += self.step_dt
        self._sample_commands(torch.nonzero(self.cmd_timer >= c.cmd_resample_s).flatten())

        # per-robot pushes: a one-off horizontal velocity kick
        self.push_timer -= self.step_dt
        ids = torch.nonzero(self.push_timer <= 0.0).flatten()
        self.push_flag.zero_()
        if len(ids):
            vel = self.robot.data.root_vel_w[ids].clone()
            ang = torch.rand(len(ids), device=self.device) * 2 * math.pi
            vel[:, 0] += c.push_vel * torch.cos(ang)
            vel[:, 1] += c.push_vel * torch.sin(ang)
            self.robot.write_root_velocity_to_sim(vel, env_ids=ids)
            self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*c.push_interval_s)
            self.push_flag[ids] = 1.0

    def _apply_action(self):
        self.robot.set_joint_position_target(self._targets)

    def _get_observations(self) -> dict:
        d, c = self.robot.data, self.cfg
        gyro = d.root_ang_vel_b
        grav = d.projected_gravity_b
        q = d.joint_pos - self.default_q
        dq = d.joint_vel
        clock, _, _ = self._gait()
        cmd = self.commands * torch.tensor([4.0, 4.0, 1.0], device=self.device)

        actor = torch.cat([gyro * 0.25, grav, cmd, q, dq * 0.05, self.actions, clock], dim=-1)
        if c.obs_noise:
            noise = torch.cat([
                torch.randn_like(gyro) * 0.1 * 0.25, torch.randn_like(grav) * 0.03, torch.zeros_like(cmd),
                torch.randn_like(q) * 0.01, torch.randn_like(dq) * 0.5 * 0.05,
                torch.zeros_like(self.actions), torch.zeros_like(clock),
            ], dim=-1)
            actor = actor + noise

        contact = self._foot_contacts().float()
        foot_h = d.body_pos_w[:, self.foot_ids, 2] - c.sole_offset
        priv = torch.cat([
            d.root_lin_vel_b * 2.0,
            (d.root_pos_w[:, 2:3] - self.stand_height),
            contact,
            foot_h * 10.0,
            self.push_flag.unsqueeze(1),
        ], dim=-1)
        return {"policy": torch.cat([actor, priv], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        c, d, dt = self.cfg, self.robot.data, self.step_dt
        lin, ang, grav = d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b
        q, dq = d.joint_pos, d.joint_vel
        contact = self._foot_contacts()
        _, stance, walking = self._gait()
        foot_h = d.body_pos_w[:, self.foot_ids, 2] - c.sole_offset
        foot_v = d.body_lin_vel_w[:, self.foot_ids, :2]
        lo, hi = d.soft_joint_pos_limits[..., 0], d.soft_joint_pos_limits[..., 1]

        t = {}
        t["track_lin"] = c.w_track_lin * torch.exp(-torch.sum((self.commands[:, :2] - lin[:, :2]) ** 2, 1) / c.sigma_lin**2)
        t["track_ang"] = c.w_track_ang * torch.exp(-((self.commands[:, 2] - ang[:, 2]) ** 2) / c.sigma_ang**2)
        t["alive"] = torch.full_like(t["track_ang"], c.w_alive)
        t["lin_vel_z"] = c.w_lin_vel_z * lin[:, 2] ** 2
        t["ang_vel_xy"] = c.w_ang_vel_xy * torch.sum(ang[:, :2] ** 2, 1)
        t["orientation"] = c.w_orientation * torch.sum(grav[:, :2] ** 2, 1)
        t["base_height"] = c.w_base_height * (d.root_pos_w[:, 2] - self.stand_height) ** 2
        t["torque"] = c.w_torque * torch.sum(d.applied_torque**2, 1)
        t["action_rate"] = c.w_action_rate * torch.sum((self.actions - self.last_actions) ** 2, 1)
        t["joint_acc"] = c.w_joint_acc * torch.sum(((dq - self.last_dq) / dt) ** 2, 1)
        t["contact_phase"] = c.w_contact_phase * (contact == stance).float().mean(1)
        t["swing_height"] = c.w_swing_height * torch.sum((foot_h - c.swing_height) ** 2 * (~stance), 1) * walking
        t["feet_slip"] = c.w_feet_slip * torch.sum(torch.sum(foot_v**2, -1) * contact, 1)
        t["hip_pose"] = c.w_hip_pose * torch.sum((q[:, self.hip_ids] - self.default_q[:, self.hip_ids]) ** 2, 1)
        t["upper_pose"] = c.w_upper_pose * torch.sum((q[:, self.upper_ids] - self.default_q[:, self.upper_ids]) ** 2, 1)
        t["joint_limit"] = c.w_joint_limit * torch.sum((lo - q).clamp(min=0) + (q - hi).clamp(min=0), 1)

        total = torch.zeros(self.num_envs, device=self.device)
        for k, v in t.items():
            v = v * dt
            total += v
            self.episode_sums[k] = self.episode_sums.get(k, torch.zeros_like(v)) + v
        total += c.termination_penalty * self.reset_terminated.float()

        self.last_actions = self.actions.clone()
        self.last_dq = dq.clone()
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        d, c = self.robot.data, self.cfg
        fallen = (d.root_pos_w[:, 2] < c.min_height_frac * self.stand_height) | (
            d.projected_gravity_b[:, :2].norm(dim=1) > c.max_tilt
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # episode logging (per-second averages of each reward term)
        if len(self.episode_sums):
            T = (self.episode_length_buf[env_ids].float() * self.step_dt).clamp(min=self.step_dt)
            log = {f"Episode_Reward/{k}": (v[env_ids] / T).mean() for k, v in self.episode_sums.items()}
            log["Episode/length_s"] = T.mean()
            log["Episode/fall_rate"] = self.reset_terminated[env_ids].float().mean()
            self.extras["log"] = log
            for v in self.episode_sums.values():
                v[env_ids] = 0.0

        super()._reset_idx(env_ids)
        k = len(env_ids)

        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 2] = self.stand_height + 0.005
        yaw = torch.rand(k, device=self.device) * 2 * math.pi - math.pi
        root[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
        root[:, 7:] = 0.0
        root[:, 7:13] += torch.randn(k, 6, device=self.device) * 0.05
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

        q = self.default_q[env_ids] + torch.randn(k, 25, device=self.device) * 0.03
        lim = self.robot.data.soft_joint_pos_limits[env_ids]
        q = torch.max(torch.min(q, lim[..., 1]), lim[..., 0])
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dq[env_ids] = 0.0
        self.push_timer[env_ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.push_interval_s)
        self.push_flag[env_ids] = 0.0
        self._sample_commands(env_ids)
        # spread episode start times so commands/pushes/timeouts aren't synchronized across robots
        if k == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, int(self.max_episode_length))
