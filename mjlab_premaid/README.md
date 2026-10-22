# Premaid AI on mjlab (GPU)

Same get-up task as `mujoco_sim/getup_env.py`, rebuilt on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp + rsl_rl) so thousands of robots run in parallel on the GPU instead of ~100 on CPU cores.

Kept identical to the CPU version, so the result runs on the real robot with `mujoco_sim/deploy`:
86-value actor observation, action scale 1.0 + 0.3 EMA filter, fitted RCB-4 servo model
(kp 43.6 / kd 0.30, ~220 deg/s slew, 0-20 ms command delay, 0-20 ms stale joint reads, ±0.02 rad offsets),
start pools (lying / squat / sit / kneel), rewards, and the 0.2 s hand-over test to the walking policy.

Differences: collision shapes are capsules/boxes (`make_robot_xml.py`; Warp is slow with meshes), robot
friction is 0.3 with the floor randomized 0.4-1.2, servo gains are randomized ±20 % per episode.

## Train (Windows PC, 3090 Ti)

1. One time, admin PowerShell: `wsl --install -d Ubuntu`, reboot, open Ubuntu once to create a user.
   (The normal Windows NVIDIA driver already exposes CUDA inside WSL2.)
2. Double-click `run_mjlab_getup.bat` in the repo root. The first run installs mjlab into
   `~/premaid_mjlab_venv` inside WSL, then trains 4096 robots for 3000 iterations.
3. Progress: `tensorboard --logdir logs/mjlab` (look at `GetUp/success_from_*`).

Linux directly: `./mjlab_premaid/train_getup.sh`.

## Watch / deploy

* `run_mjlab_export.bat` – exports the newest `model_*.pt` to `getup_policy_mjlab.pt` (TorchScript,
  normalizer baked in) and plays it in the CPU MuJoCo sim through the same loop as the real robot.
* Real robot: `python mujoco_sim/deploy/deploy_policy.py --mode getup --getup_checkpoint getup_policy_mjlab.pt`.
* mjlab's own viewer: `python -m mjlab_premaid.play Premaid-GetUp --checkpoint-file <model.pt>`.

## Files

| file | |
|---|---|
| `make_robot_xml.py` | robot-only MJCF with primitive colliders → `assets/premaid_mjlab.xml` |
| `robot.py` | mjlab entity: joints, servo actuators |
| `getup.py` | task: servo-model action term, observations, rewards, success, start poses |
| `assets/getup_pools.npz` | start poses (regenerate: `python -c "from mjlab_premaid.getup import make_pool_file; make_pool_file()"`) |
| `export.py` | rsl_rl checkpoint → TorchScript for deploy |
