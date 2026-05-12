import argparse
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# Setup App Launcher
parser = argparse.ArgumentParser("Premaid AI RL Inference (Playback)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from env_cfg import PremaidAIEnvCfg

# skrl imports
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ==============================================================================
# 1. The Phase 2 Environment (Identical to where the checkpoints were trained)
# ==============================================================================
class PremaidAIEnv(DirectRLEnv):
    cfg: PremaidAIEnvCfg

    def __init__(self, cfg: PremaidAIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.joint_targets = self.default_joint_pos.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        import isaaclab.sim as sim_utils
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        scaled_actions = actions * 1.5 
        noisy_actions = scaled_actions + torch.randn_like(scaled_actions) * 0.02
        self.joint_targets = self.default_joint_pos + noisy_actions
        
    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_targets)
        step_count = self.episode_length_buf[0]
        
        # Kept the pushes enabled so you can watch them recover!
        if step_count > 0 and torch.rand(1, device=self.device).item() < 0.01:
            push_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
            push_forces[:, 0, 0:2] = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 40.0
            push_torques = torch.zeros_like(push_forces)
            self.robot.set_external_force_and_torque(forces=push_forces, torques=push_torques)

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        bno055_accel = self.robot.data.projected_gravity_b 
        bno055_gyro = self.robot.data.root_ang_vel_b
        base_lin_vel = self.robot.data.root_lin_vel_b

        obs = torch.cat([
            joint_pos + torch.randn_like(joint_pos) * 0.01,
            joint_vel + torch.randn_like(joint_vel) * 0.1,
            bno055_accel + torch.randn_like(bno055_accel) * 0.02,
            bno055_gyro + torch.randn_like(bno055_gyro) * 0.003, 
            base_lin_vel 
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Rewards don't matter during playback, but the function must return a tensor
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fallen = self.robot.data.root_pos_w[:, 2] < 0.13
        time_out = self.episode_length_buf >= self.max_episode_length
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids] 
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        self.joint_targets[env_ids] = default_joint_pos.clone()

# ==============================================================================
# 2. Neural Networks
# ==============================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1)
        )
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ==============================================================================
# 3. Playback Loop
# ==============================================================================
def main():
    env_cfg = PremaidAIEnvCfg()
    # env_cfg.scene.num_envs = 4

    env = PremaidAIEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env) 

    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device) 
    }

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(models=models, memory=None, cfg=cfg_ppo, observation_space=env.observation_space, action_space=env.action_space, device=env.device)

    # ==========================================================================
    # PASTE YOUR CHECKPOINT PATH HERE TO TEST IT
    # ==========================================================================
    CHECKPOINT_PATH = "runs/premaid_ai/26-05-12_14-30-56-504792_PPO/checkpoints/agent_13000.pt"
    
    print(f"[INFO] Loading Phase 2 Checkpoint: {CHECKPOINT_PATH}")
    agent.load(CHECKPOINT_PATH)
    
    # THIS IS THE MAGIC LINE: Tells the brain to stop learning and exploring
    agent.set_mode("eval")

    print("[INFO] Spawning robots. Press Ctrl+C to stop.")
    obs, _ = env.reset()
    
    # Infinite visual loop
    while simulation_app.is_running():
        with torch.no_grad():
            # Get exact deterministic actions
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
            
        obs, reward, terminated, truncated, info = env.step(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()