# train_direct.py
import argparse
import torch
import torch.nn as nn

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
        # Scale actions
        scaled_actions = actions * 0.5 
        
        # ADD ACTION NOISE: Simulate backlash in the 3D printed gears
        noisy_actions = scaled_actions + torch.randn_like(scaled_actions) * 0.02
        
        # CALCULATE TARGETS
        self.joint_targets = self.default_joint_pos + noisy_actions
        
    def _apply_action(self):
        # 1. Send commands to motors
        self.robot.set_joint_position_target(self.joint_targets)
        
        # 2. THE RANDOM PUSH:
        # Every ~500 steps, apply a random physical force to the robot's base
        # to simulate tripping, uneven floors, or wind.
        if self.episode_length_buf[0] % 500 == 0:
            # Create a random push force between -50N and +50N on X and Y axes
            push_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
            push_forces[:, 0, 0:2] = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 20.0
            
            push_torques = torch.zeros_like(push_forces)
            
            # Apply both to the root body
            self.robot.set_external_force_and_torque(forces=push_forces, torques=push_torques)

    def _get_observations(self) -> dict:
        # Get the perfect math data
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        imu_data = self.robot.data.projected_gravity_b
        
        # ADD NOISE: Simulate cheap sensors by adding random jitter
        # +/- 0.01 radians for position, +/- 0.1 for velocity, +/- 0.05 for IMU
        obs = torch.cat([
            joint_pos + torch.randn_like(joint_pos) * 0.01,
            joint_vel + torch.randn_like(joint_vel) * 0.1,
            imu_data + torch.randn_like(imu_data) * 0.05, 
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. The Main Goal: Move Forward (X-axis)
        forward_vel = self.robot.data.root_lin_vel_w[:, 0]
        reward = forward_vel * 2.0 
        
        # 2. Posture Penalty: Penalize leaning forward/backward or sideways
        # projected_gravity_b is [X, Y, Z]. If the robot is perfectly upright, X and Y are 0.
        pitch_roll_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        reward -= pitch_roll_error * 5.0 # Heavy penalty for leaning!

        # 3. Energy Penalty: Penalize moving the motors too fast (jittering)
        joint_velocities = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        reward -= joint_velocities * 0.01 
        
        # 4. Action Rate Penalty: Penalize suddenly changing the motor direction
        # (You would need to store 'self.previous_actions' in _pre_physics_step to calculate this)
        
        # 5. The Death Penalty: Torso drops below 20cm
        fallen = self.robot.data.root_pos_w[:, 2] < 0.2
        reward[fallen] -= 10.0
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        fallen = self.robot.data.root_pos_w[:, 2] < 0.2
        time_out = self.episode_length_buf >= self.max_episode_length
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 1. Let the base class reset the episode timers
        super()._reset_idx(env_ids)
        
        # If env_ids is None, the simulator is asking to reset ALL robots
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        # 2. Pick the robot up and put it back in the air
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        # Shift them so they spawn on their specific grid squares, not all in the center!
        default_root_state[:, :3] += self.scene.env_origins[env_ids] 
        
        # Write the new position and kill any falling momentum (velocity)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        
        # 3. Snap the joints back to the default T-pose
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        
        # 4. Reset our target tracking so it doesn't immediately try to resume its fallen pose
        self.joint_targets[env_ids] = default_joint_pos.clone()
        if hasattr(self, "previous_actions"):
            self.previous_actions[env_ids] = 0.0

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
    # Initialize the Environment
    env_cfg = PremaidAIEnvCfg()
    env = PremaidAIEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env) # Wrap for skrl

    # Initialize Networks
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device)
    }

    # PPO Configuration
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 64
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 4
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 1e-3
    cfg_ppo["experiment"]["write_interval"] = 100
    cfg_ppo["experiment"]["directory"] = "runs/premaid_ai"
    cfg_ppo["experiment"]["checkpoint_interval"] = 1000  # Save every 1,000 PPO updates
    cfg_ppo["experiment"]["store_separately"] = False    # Overwrite the old save to save disk space

    memory = RandomMemory(memory_size=cfg_ppo["rollouts"], num_envs=env.num_envs, device=env.device)
    agent = PPO(models=models, memory=memory, cfg=cfg_ppo, observation_space=env.observation_space, action_space=env.action_space, device=env.device)
    
    # RESUME FROM CHECKPOINT (Uncomment this line if you crash and need to resume)
    # agent.load("runs/premaid_ai/agent.pt")

    # Start Training
    print("[INFO]: Launching skrl PPO Training...")
    cfg_trainer = {"timesteps": 5000000, "headless": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    trainer.train()
    simulation_app.close()

if __name__ == "__main__":
    main()