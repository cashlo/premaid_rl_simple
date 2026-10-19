# Premaid AI walking — MuJoCo version

A MuJoCo port of the Isaac Lab walking setup in the repo root. It runs on any
machine with a CPU (no Isaac Sim / Omniverse needed) and trains with the same
library (skrl PPO).

```
mujoco_sim/
  assets/premaidai.xml        MJCF model (25 servos, 1.80 kg, generated)
  assets/meshes/*.stl         visual + collision meshes (Apache-2.0, see LICENSE.txt)
  premaid_env.py              walking task: observations, rewards, randomization
  vec_env.py                  multi-process batching + skrl wrapper
  train_mujoco.py             PPO training
  play_mujoco.py              viewer with keyboard teleop / mp4 recording / TorchScript export
  tools/build_mjcf.py         regenerates the model from the URDF source
```

## Setup

```bash
pip install mujoco skrl torch gymnasium imageio imageio-ffmpeg
```

## Train

**Windows one-click:** double-click `mujoco_sim\run_training_windows.bat`. It pulls this folder from the
`mujoco-walking` branch, creates a `premaid_mujoco` conda env on first run (CUDA torch, mujoco, skrl),
and trains with (cores - 2) worker processes, resuming the newest checkpoint if one exists.

```bash
cd mujoco_sim
python train_mujoco.py                              # 256 robots on (cores-1) processes
python train_mujoco.py --num_envs 512 --workers 15  # bigger machine
python train_mujoco.py --resume latest              # continue the newest run
tensorboard --logdir runs                           # Episode_Reward/* curves per reward term
```

Throughput is roughly 1,500 robot-steps/s per CPU core (MuJoCo itself is ~40 µs
per physics step; the rest is Python). A 16-core machine does ~20k steps/s, so
50M samples (where a clean gait usually appears) takes about 45 minutes. The
learner uses CUDA automatically when available; the bottleneck is CPU stepping.

Watch `Episode_Reward/track_lin` (max 2.0) and `Episode/length_s` (max 20).
Standing comes first, then shuffling, then stepping with the gait clock.

## Play

```bash
python play_mujoco.py                                   # newest checkpoint, interactive viewer
python play_mujoco.py --video walk.mp4 --cmd 0.12 0 0   # headless mp4 (set MUJOCO_GL=egl or osmesa on servers)
python play_mujoco.py --export premaid_policy.pt        # TorchScript: 86 obs -> 25 actions
```

Viewer keys: ↑/↓ forward/back, ←/→ turn, `,`/`.` strafe, `K` stop, Backspace reset.

## The model

The original `premaidai.usd` in this repo only references `configuration/*.usd`
files that aren't committed, so the robot was rebuilt from
[chikuta/premaidai_description](https://github.com/chikuta/premaidai_description)
(kinematics, masses, inertias, meshes by kuroiwasi, Apache-2.0):

* 25 position-controlled joints, **actuators in servo-ID order** (`SERVO_ORDER`
  in `premaid_env.py`, from `premaidai_controller/config/joint_config.yaml`), so
  the policy's action vector maps 1:1 onto servo IDs 2–28.
* Servo torque limits from the URDF: 1.37 N·m legs, 0.91 N·m arms/head.
  Position gains kp = 6 (legs) / 3 (arms), kv = 0.08, armature 0.002. These are
  guesses, not measurements — randomized ±20% during training. Tune them in
  `tools/build_mjcf.py` once you have servo step-response data.
* The URDF inertias are labelled `e-5` but are really in units of 1e-7 kg·m²
  (checked against each mesh's convex hull); the builder fixes that.
* Collisions: flat box soles for the feet, convex hulls for the rest. Legs
  collide with each other; body/arms/head only with the floor (touching the
  floor with anything but a sole ends the episode).

Regenerate after editing gains/limits:

```bash
git clone https://github.com/chikuta/premaidai_description /tmp/premaidai_description
pip install trimesh pycollada
python tools/build_mjcf.py --description /tmp/premaidai_description
```

## Task design (vs. the Isaac Lab version)

| | Isaac Lab (`train_direct.py`) | MuJoCo (`premaid_env.py`) |
|---|---|---|
| Control rate | 30 Hz | 50 Hz (sim 200 Hz) |
| Action | `default + 1.5 * a` | `default + 0.30 * a` legs, `0.20 * a` arms/head |
| Default pose | all zeros (straight legs) | slight crouch, arms down |
| Commands | vx 0–1 m/s | vx −0.10–0.20, vy ±0.08 m/s, yaw ±0.6 rad/s, 15% "stand" |
| Actor obs | joints, IMU, **base linear velocity**, command | joints, IMU, command, last action, gait clock (no base velocity — the real robot can't measure it) |
| Critic | same as actor | privileged: + base velocity, height, foot contacts, friction |
| Rewards | tracking, posture, energy, fall | + gait-phase foot contacts, swing height, slip, action rate, hip/arm pose, joint limits |
| Randomization | pushes, action/obs noise | friction, torso mass, servo gains, damping, velocity pushes, obs noise |

Command speeds are scaled to the robot: it is 0.38 m tall, so 1 m/s would be
several body lengths per second. Widen `cmd_vx` in `WalkCfg` once it walks.
