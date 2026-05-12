import argparse
import torch
import torch.nn as nn
import os
import glob

from isaaclab.app import AppLauncher

# Setup App Launcher (Must be done before other heavy imports)
parser = argparse.ArgumentParser("Premaid AI RL Training with skrl")
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
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ==============================================================================
# 1. The Custom Environment Class
# ==============================================================================
class PremaidAIEnv(DirectRLEnv):
    cfg: PremaidAIEnvCfg

    def __init__(self, cfg: PremaidAIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.joint_targets = self.default_joint_pos.clone()
        
        # --- NEW: Initialize the Command Buffer (vx, vy, yaw) ---
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # Add ground plane and lights
        import isaaclab.sim as sim_utils
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Increased action scale to give the robot full range of motion (±1.5 radians)
        scaled_actions = actions * 1.5 
        
        noisy_actions = scaled_actions + torch.randn_like(scaled_actions) * 0.02
        self.joint_targets = self.default_joint_pos + noisy_actions
        
    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_targets)
        
        step_count = self.episode_length_buf[0]

        # --- NEW: Resample commands every 150 steps (5 seconds) ---
        if step_count > 0 and step_count % 150 == 0:
            # Forward speed: 0.0 to 1.0 m/s
            self.commands[:, 0] = torch.rand(self.num_envs, device=self.device) * 1.0
            # Strafing speed: -0.25 to 0.25 m/s
            self.commands[:, 1] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 1.0
            # Turning speed: -0.5 to 0.5 rad/s
            self.commands[:, 2] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 2.0

        # THE RANDOM PUSH: Kept for continuous robustness
        if step_count > 0 and torch.rand(1, device=self.device).item() < 0.01:
            push_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
            push_forces[:, 0, 0:2] = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 4.0
            push_torques = torch.zeros_like(push_forces)
            self.robot.set_external_force_and_torque(forces=push_forces, torques=push_torques)

    def _get_observations(self) -> dict:
        # 1. Joint States
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # 2. BNO055 IMU Simulation
        bno055_accel = self.robot.data.projected_gravity_b 
        bno055_gyro = self.robot.data.root_ang_vel_b
        
        # 3. Base Velocity 
        base_lin_vel = self.robot.data.root_lin_vel_b

        obs = torch.cat([
            joint_pos + torch.randn_like(joint_pos) * 0.01,
            joint_vel + torch.randn_like(joint_vel) * 0.1,
            bno055_accel + torch.randn_like(bno055_accel) * 0.02,
            bno055_gyro + torch.randn_like(bno055_gyro) * 0.003, 
            base_lin_vel,
            self.commands  # --- NEW: Added Joystick commands to the brain ---
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.ones(self.num_envs, device=self.device) * 0.2

        # --- NEW: 1. Tracking Rewards ---
        # Calculate the mathematical difference between what we commanded and what the robot is actually doing
        lin_vel_error_x = torch.square(self.commands[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        lin_vel_error_y = torch.square(self.commands[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
        ang_vel_error_z = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])

        # Torch.exp() makes the reward drop off quickly if the error gets large.
        reward += torch.exp(-lin_vel_error_x / 1.0) * 2.0  # Highly weight forward tracking
        reward += torch.exp(-lin_vel_error_y / 1.0) * 1.0  # Strafe tracking
        reward += torch.exp(-ang_vel_error_z / 1.0) * 1.0  # Turn tracking
        
        # 2. Posture Penalty
        pitch_roll_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        reward -= pitch_roll_error * 5.0 

        # 3. Energy Penalty
        joint_velocities = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        reward -= joint_velocities * 0.01 
        
        # 4. The Death Penalty
        fallen = self.robot.data.root_pos_w[:, 2] < 0.13
        reward[fallen] -= 50.0
        
        return reward

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
        if hasattr(self, "previous_actions"):
            self.previous_actions[env_ids] = 0.0

        # --- NEW: Assign brand new random commands when the robot resets ---
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 1.0
        self.commands[env_ids, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0
        self.commands[env_ids, 2] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2.0


# ==============================================================================
# 2. Define the Neural Networks
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
# 3. Execution Main Loop
# ==============================================================================
def main():
    env_cfg = PremaidAIEnvCfg()
    env = PremaidAIEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env) 

    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device)
    }

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 128  # Increased to give the agent longer continuous trajectories
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 4
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 5e-5  # --- UPDATED: The "7-Iron" for Network Surgery ---
    # cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    # cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

    cfg_ppo["experiment"]["write_interval"] = 100
    cfg_ppo["experiment"]["directory"] = "runs/premaid_ai"
    cfg_ppo["experiment"]["checkpoint_interval"] = 1000 
    cfg_ppo["experiment"]["store_separately"] = False   

    memory = RandomMemory(memory_size=cfg_ppo["rollouts"], num_envs=env.num_envs, device=env.device)
    
    model_upgrade = True
    if model_upgrade:
        # Load your exact Golden Checkpoint
        checkpoint_path = "runs/premaid_ai/26-05-11_20-50-42-412142_PPO/checkpoints/agent_1000.pt"
        print(f"[INFO] Performing Network Surgery on {checkpoint_path}...")
        
        # 1. Load the raw dictionary from the hard drive
        checkpoint = torch.load(checkpoint_path, map_location=env.device)
        
        for model_name in ["policy", "value"]:
            old_state_dict = checkpoint[model_name]
            new_state_dict = models[model_name].state_dict()
            
            # 2. Grab the old weights from the very first layer
            old_weight = old_state_dict['net.0.weight']
            
            # 3. Create a new empty matrix of the correct size [256, 62] full of zeros
            new_weight = torch.zeros_like(new_state_dict['net.0.weight'])
            
            # 4. Copy the old muscle memory into the first 59 slots
            new_weight[:, :old_weight.shape[1]] = old_weight
            
            # 5. Overwrite the state dict with our upgraded matrix
            old_state_dict['net.0.weight'] = new_weight

            if model_name == "policy":
                old_state_dict['log_std_parameter'] = torch.zeros_like(new_state_dict['log_std_parameter'])
            
            # 6. Load the hacked brain directly into the model
            models[model_name].load_state_dict(old_state_dict, strict=False)

        print("[INFO] Surgery complete! Brain successfully upgraded to 62 inputs.")
        # =====================================================================

        # Initialize the agent
        agent = PPO(models=models, memory=memory, cfg=cfg_ppo, observation_space=env.observation_space, action_space=env.action_space, device=env.device)
    else:
        search_path = os.path.join("runs", "premaid_ai", "*", "checkpoints", "*.pt")
        all_checkpoints = glob.glob(search_path)
        
        if not all_checkpoints:
            raise FileNotFoundError("Could not find any checkpoints in runs/premaid_ai/")
            
        # Sort by file modification time to get the absolute newest file
        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        checkpoint_path = latest_checkpoint
        
        print(f"[INFO] Auto-loaded latest checkpoint: {checkpoint_path}")
        
        agent = PPO(models=models, memory=memory, cfg=cfg_ppo, observation_space=env.observation_space, action_space=env.action_space, device=env.device)
        agent.load(checkpoint_path)

    print("[INFO]: Launching skrl PPO Training...")
    cfg_trainer = {"timesteps": 5000000, "headless": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    trainer.train()
    simulation_app.close()

if __name__ == "__main__":
    main()