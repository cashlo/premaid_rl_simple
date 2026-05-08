import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np

# Rcb4Lib is expected to be in the same directory or in PYTHONPATH
try:
    from Rcb4BaseLib import Rcb4BaseLib
except ImportError:
    # Add Rcb4Lib to path if not found directly
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Rcb4Lib'))
    from Rcb4BaseLib import Rcb4BaseLib

# Define servo mappings
# 25 joints, assuming IDs 1-25 on SIO 1 for simplicity (needs real mapping for production)
NUM_JOINTS = 25
SERVO_IDS = [i for i in range(1, NUM_JOINTS + 1)]
SERVO_SIOS = [1] * NUM_JOINTS # Using SIO1 for all as a default

# ==============================================================================
# 2. Define the Neural Network (Mirrors train_direct.py)
# ==============================================================================
# We define a standalone PyTorch Policy class here to avoid importing
# the IsaacLab simulator environment from train_direct.py

from skrl.models.torch import Model, GaussianMixin
import gymnasium as gym

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        # We pass observation_space and action_space from gym spaces conceptually
        # Here we just pass the raw dimensions for standalone usage
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space,))
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_space,))

        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

# ==============================================================================
# 3. Execution Control Loop
# ==============================================================================

def main():
    print("Initializing RCB-4HV connection...")
    rcb4 = Rcb4BaseLib()

    # Open connection to RCB-4HV via Dual USB adapter HS
    # Try different common ports
    connected = False
    for port in ['/dev/ttyUSB0', '/dev/ttyAMA0']:
        try:
            print(f"Trying port {port}...")
            # portName, baudrate, timeout(s)
            rcb4.open(port, 115200, 1.3)
            if rcb4.checkAcknowledge():
                print(f"Successfully connected to RCB-4HV on {port}!")
                print(f"RCB-4HV Version: {rcb4.Version}")
                connected = True
                break
            else:
                rcb4.close()
        except Exception as e:
            print(f"Failed to connect on {port}: {e}")

    if not connected:
        print("Failed to connect to RCB-4HV. Please check the connection.")
        return

    # Initialize PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch device: {device}")

    # Load the trained policy
    # Observation space: 53 (25 joint pos + 25 joint vel + 3 IMU projected gravity)
    # Action space: 25
    print("Loading trained policy...")
    policy = Policy(observation_space=53, action_space=25, device=device)
    policy.to(device)

    # In a real scenario, you'd load weights:
    # policy.load("runs/premaid_ai/checkpoints/agent.pt")
    policy.eval()

    # Hardware conversion constants
    # (Assuming 8000 is center, steps per radian)
    STEPS_PER_RADIAN = 1500.0 # Just a placeholder value
    RCB4_CENTER_POS = 8000

    # Initialize state variables
    last_joint_pos_rad = np.zeros(NUM_JOINTS)
    last_time = time.time()

    # Assume default T-pose is 0 rad for all joints conceptually
    default_joint_pos_rad = np.zeros(NUM_JOINTS)

    print("Starting control loop. Press Ctrl+C to stop.")

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.01 # Prevent divide by zero

            # --- 1. Get Observations ---
            current_joint_pos_steps = []
            for i in range(NUM_JOINTS):
                sid = SERVO_IDS[i]
                sio = SERVO_SIOS[i]
                success, pos_step = rcb4.getSinglePos(sid, sio)
                if not success:
                    # Fallback to center if read fails
                    pos_step = RCB4_CENTER_POS
                current_joint_pos_steps.append(pos_step)

            # Convert steps to radians
            # Assuming: steps = center + (rad * steps_per_rad)
            # rad = (steps - center) / steps_per_rad
            current_joint_pos_rad = (np.array(current_joint_pos_steps) - RCB4_CENTER_POS) / STEPS_PER_RADIAN

            # Calculate velocity
            current_joint_vel_rad = (current_joint_pos_rad - last_joint_pos_rad) / dt

            # Mock IMU data (since IMU on ADC is unconfirmed, we use perfect upright zero vector)
            mock_imu_projected_gravity = np.array([0.0, 0.0, -1.0])

            # Construct observation tensor (must match _get_observations in env)
            obs_array = np.concatenate([
                current_joint_pos_rad,
                current_joint_vel_rad,
                mock_imu_projected_gravity
            ]).astype(np.float32)

            obs_tensor = torch.tensor(obs_array, device=device).unsqueeze(0) # Add batch dim

            # --- 2. Run Policy ---
            with torch.no_grad():
                actions, _, _ = policy.compute({"states": obs_tensor}, role="policy")

            actions_np = actions.squeeze(0).cpu().numpy()

            # --- 3. Apply Actions ---
            # Environment scaled actions: scaled_actions = actions * 0.5
            # Targets: joint_targets = default_joint_pos + scaled_actions
            scaled_actions = actions_np * 0.5
            target_joint_pos_rad = default_joint_pos_rad + scaled_actions

            # Convert target radians back to RCB-4HV steps
            target_joint_pos_steps = (target_joint_pos_rad * STEPS_PER_RADIAN) + RCB4_CENTER_POS
            target_joint_pos_steps = np.clip(target_joint_pos_steps, 0, 16000).astype(int) # RCB-4HV valid range

            # Send commands to servos
            # Use runSingleServoCmd for each servo
            # Note: frame=1 means fastest possible movement for this command slice
            for i in range(NUM_JOINTS):
                sid = SERVO_IDS[i]
                sio = SERVO_SIOS[i]
                pos = int(target_joint_pos_steps[i])
                rcb4.setSingleServo(sid, sio, pos, 1)

            # Update state variables
            last_joint_pos_rad = current_joint_pos_rad
            last_time = current_time

            # Sleep slightly to maintain a control frequency
            # (e.g. env_cfg.py decimation=2 means ~50Hz assuming 100Hz base sim, adjust as needed)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nControl loop stopped by user.")
    finally:
        print("Freeing servos and closing connection...")
        # Free servos (0x8000)
        for i in range(NUM_JOINTS):
            rcb4.setFreeSingleServo(SERVO_IDS[i], SERVO_SIOS[i])
        rcb4.close()

if __name__ == "__main__":
    main()
