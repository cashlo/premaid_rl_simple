import argparse
import os
import glob
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# ==============================================================================
# Setup App Launcher (Must be at the very top!)
# ==============================================================================
parser = argparse.ArgumentParser("Premaid AI RL Inference (Playback)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of robots to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# Isaac Lab Imports (Must be imported AFTER AppLauncher)
# ==============================================================================
# --- FIXED: The new Isaac Lab path for the keyboard ---
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from env_cfg import PremaidAIEnvCfg

# SKRL imports
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ==============================================================================
# 1. The Phase 3 Environment 
# ==============================================================================
class PremaidAIEnv(DirectRLEnv):
    cfg: PremaidAIEnvCfg

    def __init__(self, cfg: PremaidAIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.joint_targets = self.default_joint_pos.clone()
        
        # Command Buffer
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)

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
        
        # The Ghost Pushes (4.0 Newtons for robustness)
        if step_count > 0 and torch.rand(1, device=self.device).item() < 0.01:
            push_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
            push_forces[:, 0, 0:2] = (torch.rand(self.num_envs, 2, device=self.device) - 0.5) * 4.0
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
            base_lin_vel,
            self.commands # Policy sees the joystick!
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
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
        
        # Reset command buffer to 0 when falling
        self.commands[env_ids] = 0.0 

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
# 3. Playback & Teleop Loop
# ==============================================================================
def main():
    env_cfg = PremaidAIEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = PremaidAIEnv(cfg=env_cfg)
    env_wrapped = SkrlVecEnvWrapper(env) 

    models = {
        "policy": Policy(env_wrapped.observation_space, env_wrapped.action_space, env_wrapped.device),
        "value": Value(env_wrapped.observation_space, env_wrapped.action_space, env_wrapped.device) 
    }

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    agent = PPO(models=models, memory=None, cfg=cfg_ppo, 
                observation_space=env_wrapped.observation_space, 
                action_space=env_wrapped.action_space, 
                device=env_wrapped.device)

    # ==========================================================================
    # AUTO-FIND LATEST CHECKPOINT
    # ==========================================================================
    search_path = os.path.join("runs", "premaid_ai", "*", "checkpoints", "*.pt")
    all_checkpoints = glob.glob(search_path)
    
    if not all_checkpoints:
        raise FileNotFoundError("Could not find any checkpoints in runs/premaid_ai/")
        
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    latest_checkpoint = "runs/premaid_ai/26-05-12_23-27-53-411878_PPO/checkpoints/agent_100000.pt"
    print(f"\n[INFO] Auto-Loading Latest Checkpoint: {latest_checkpoint}")
    
    agent.load(latest_checkpoint)
    agent.set_mode("eval")

    # ==========================================================================
    # INITIALIZE KEYBOARD TELEOP
    # ==========================================================================
    
    teleop_interface = Se2Keyboard(Se2KeyboardCfg())
    
    print("\n" + "="*50)
    print("🎮 CONTROLS ACTIVE!")
    print("Click the Omniverse Viewport to take control.")
    print("W / S : Walk Forward / Backward")
    print("A / D : Strafe Left / Right")
    print("Q / E : Turn Left / Right")
    print("="*50 + "\n")

    obs, _ = env_wrapped.reset()
    
    while simulation_app.is_running():
        
        # 1. Read Keyboard Input
        teleop_cmd = teleop_interface.advance()

        if teleop_cmd[0] != 0 or teleop_cmd[1] != 0 or teleop_cmd[2] != 0:
            print(f"🎮 Joystick Command: {teleop_cmd}")

        cmd_tensor = torch.tensor(teleop_cmd, device=env_wrapped.device, dtype=torch.float32)
        
        # 2. Overwrite the command buffer for ALL robots
        env_wrapped.unwrapped.commands[:, 0] = cmd_tensor[0]
        env_wrapped.unwrapped.commands[:, 1] = cmd_tensor[1]
        env_wrapped.unwrapped.commands[:, 2] = cmd_tensor[2]

        # 3. Standard Step
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
            
        obs, reward, terminated, truncated, info = env_wrapped.step(actions)

    simulation_app.close()

if __name__ == "__main__":
    main()