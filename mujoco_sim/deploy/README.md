# Running the policy on the real Premaid AI

`deploy_policy.py` reads the RCB-4 (servo positions) and the backpack IMU, builds exactly the
86-value observation the policy was trained on, and sends 25 servo targets at 50 Hz.
Double-click `run_deploy.bat` (in `mujoco_sim`) or call it with arguments, e.g.
`run_deploy.bat --mode check`. Go through the modes in order the first time:

| step | command | torque | what to look for |
|---|---|---|---|
| 1 | `--mode scan` | unchanged | every joint in `servo_map.py` is found. IDs above 17 can't be addressed on an RCB-4 SIO group, so some may need new (id, sio) values |
| 2 | `--mode check` | **off** | move each joint by hand; the printed angle must follow the simulator's convention (default crouch = knees +40°, hips −20°, ankles −20°). Upright → gravity ≈ (0, 0, −1); tilt forward → x > 0; lean left → y > 0 |
| 3 | `--mode bench` | off | joint reads + writes must fit in 20 ms. At 115200 baud they won't: set the RCB-4 COM speed to 1.25 Mbps and pass `--rcb4_baud 1250000`, and set the USB-serial latency timer to 1 ms |
| 4 | `--mode stand` | **on** | robot eases into the crouch over 2 s and holds it |
| 5 | `--mode run --checkpoint <agent_N.pt>` | **on** | in a harness first. Keys: w/s forward/back, a/d strafe, q/e turn, space stop, x quit |

Ctrl+C, `x`, or a fall (> ~45° tilt) makes every servo go limp.

`--sim` runs stand/run against the MuJoCo robot (same servo model and delays as training), which is
how this script was tested: it walks 1.09 m in 10 s at a 0.12 m/s command, matching the training
evaluation. `--sim --render` shows it.

Ports default to `--rcb4 COM5 --imu COM4 --imu_baud 1000000` (your existing scripts).
