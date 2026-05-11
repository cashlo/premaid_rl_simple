# env_cfg.py
import torch
import isaaclab.sim as sim_utils

# Import the new requirements
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

            # The PD Gains
            stiffness=2.5,  # Strong enough to hold weight, soft enough to absorb impact
            damping=0.15,   # Just enough friction to stop jitters
            
            # Hardware Limits (Prevents physics explosions and AI cheating)
            effort_limit=1.6,   # Caps the max torque to the real-world limit (N·m)
            velocity_limit=8.0, # Caps the max joint speed (rad/s)  
        ),
    },
)

# ==============================================================================
# Environment Configuration
# ==============================================================================
@configclass # <--- THIS DECORATOR IS MANDATORY NOW
class PremaidAIEnvCfg(DirectRLEnvCfg):
    # 1. Environment Settings
    env_name = "PremaidAI-Walk-v0"
    decimation = 2 
    render_interval = 2
    episode_length_s = 10.0 
    
    # 2. RL Space Definition (Renamed in the new update)
    action_space = 25 
    observation_space = 53 
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=512 * 1024,
            gpu_max_rigid_contact_count=2048 * 1024,
            gpu_found_lost_pairs_capacity=512 * 1024,
            # It's good practice to ensure these are enabled for GPU pipelines
            # use_gpu=True,
            # enable_pcm=True,
        )
    )
    
    # 3. Scene Setup (num_envs and spacing moved here)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, 
        # num_envs=4096,
        env_spacing=1.5
    )
    
    # 4. Robot Definition (Must include the prim_path cloning logic)
    robot: ArticulationCfg = PREMAID_AI_CFG.replace(prim_path="/World/envs/env_.*/Robot")