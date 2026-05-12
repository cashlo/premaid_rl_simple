import torch
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ==============================================================================
# Hardware Configuration
# ==============================================================================
PREMAID_AI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="premaidai.usd", 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3), 
        joint_pos={".*": 0.0},
    ),
    actuators={
        "kondo_servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"], 
            stiffness=2.5,
            damping=0.15,
            effort_limit=1.6,
            velocity_limit=8.0,
        ),
    },
)

# ==============================================================================
# Environment Configuration
# ==============================================================================
@configclass 
class PremaidAIEnvCfg(DirectRLEnvCfg):
    # 1. Environment Settings
    env_name = "PremaidAI-Walk-v0"
    decimation = 2 
    render_interval = 2
    episode_length_s = 10.0 
    
    # 2. RL Space Definition
    action_space = 25 
    observation_space = 62  # 25 (pos) + 25 (vel) + 3 (BNO055_accel) + 3 (BNO055_gyro) + 3 (lin_vel)
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=512 * 1024,
            gpu_max_rigid_contact_count=2048 * 1024,
            gpu_found_lost_pairs_capacity=512 * 1024,
        )
    )
    
    # 3. Scene Setup
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # Halved from 8192 to balance the PPO batch size
        env_spacing=1.5
    )
    
    # 4. Robot Definition
    robot: ArticulationCfg = PREMAID_AI_CFG.replace(prim_path="/World/envs/env_.*/Robot")