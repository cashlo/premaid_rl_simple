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
        pos=(0.0, 0.0, 0.4), 
        joint_pos={".*": 0.0},
    ),
    actuators={
        "kondo_servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"], 
            stiffness=400.0, 
            damping=40.0,    
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
    episode_length_s = 10.0 
    
    # 2. RL Space Definition (Renamed in the new update)
    action_space = 25 
    observation_space = 53 
    state_space = 0
    
    # 3. Scene Setup (num_envs and spacing moved here)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, 
        env_spacing=1.5
    )
    
    # 4. Robot Definition (Must include the prim_path cloning logic)
    robot: ArticulationCfg = PREMAID_AI_CFG.replace(prim_path="/World/envs/env_.*/Robot")