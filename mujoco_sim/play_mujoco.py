"""Watch / drive / export a trained Premaid AI walking policy in MuJoCo.

    python play_mujoco.py                              # interactive viewer, newest checkpoint
    python play_mujoco.py --checkpoint path/to/agent_5000.pt
    python play_mujoco.py --video walk.mp4 --cmd 0.12 0 0 --seconds 10     # headless video
    python play_mujoco.py --export premaid_policy.pt   # TorchScript (obs -> action) for deployment

Viewer keys (click the viewer window first):
    Up / Down      forward / backward speed
    Left / Right   turn left / right
    , / .          strafe left / right
    K              stop (zero command)
    Backspace      reset the robot
"""

from __future__ import annotations

import argparse
import threading
import time
import types

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (import before mujoco: avoids a segfault with MUJOCO_GL=osmesa)

from premaid_env import NUM_OBS, PremaidWalkEnv, WalkCfg
from train_mujoco import find_checkpoint, make_agent

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_BACKSPACE = 265, 264, 263, 262, 259


def load_agent(path):
    import gymnasium

    from premaid_env import NUM_ACTIONS, NUM_STATES

    fake = types.SimpleNamespace(
        observation_space=gymnasium.spaces.Box(-np.inf, np.inf, (NUM_OBS,), np.float32),
        state_space=gymnasium.spaces.Box(-np.inf, np.inf, (NUM_STATES,), np.float32),
        action_space=gymnasium.spaces.Box(-1, 1, (NUM_ACTIONS,), np.float32),
        num_envs=1,
    )
    agent = make_agent(fake, "cpu")
    agent.load(path)
    agent.enable_training_mode(False, apply_to_models=True)
    return agent


class DeterministicPolicy(torch.nn.Module):
    """obs (raw, as produced by PremaidWalkEnv) -> action, with the running normalizer baked in."""

    def __init__(self, agent):
        super().__init__()
        pre = agent._observation_preprocessor
        self.register_buffer("mean", pre.running_mean.clone().float())
        self.register_buffer("var", pre.running_variance.clone().float())
        self.clip = float(pre.clip_threshold)
        self.eps = float(pre.epsilon)
        self.net = agent.policy.net

    def forward(self, obs):
        # same formula as skrl's RunningStandardScaler
        x = torch.clamp((obs - self.mean) / (torch.sqrt(self.var) + self.eps), -self.clip, self.clip)
        return self.net(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--video", default=None, help="write an mp4 instead of opening the viewer")
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.12, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--export", default=None, help="save a TorchScript policy and exit")
    ap.add_argument("--randomize", action="store_true", help="keep domain randomization + pushes on")
    ap.add_argument("--debris", type=float, default=0.0, help="floor debris level 0 (flat) .. 1 (hardest)")
    args = ap.parse_args()

    path = find_checkpoint(args.checkpoint)
    print(f"[INFO] loading {path}")
    policy = DeterministicPolicy(load_agent(path)).eval()

    if args.export:
        torch.jit.script(policy).save(args.export)
        print(f"[INFO] exported TorchScript policy to {args.export} (input: {NUM_OBS} floats, output: 25 actions)")
        print("       joint target = default_pose + action * scale (0.30 legs / 0.20 arms+head), servo-ID order")
        return

    env = PremaidWalkEnv(
        WalkCfg(randomize=args.randomize, obs_noise=args.randomize, debris_level_init=args.debris, debris_level_step=0.0),
        seed=0,
    )
    env.manual_commands = True
    env.commands[:] = args.cmd
    obs, _ = env.reset()

    def act(o):
        with torch.no_grad():
            return policy(torch.as_tensor(o).unsqueeze(0))[0].numpy()

    if args.video:
        record(env, act, obs, args)
    else:
        interactive(env, act, obs)


def record(env, act, obs, args):
    import imageio
    import mujoco

    m, d = env.model, env.data
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.base_id
    cam.distance, cam.elevation, cam.azimuth = 1.0, -15, 120
    fps = int(round(1 / env.dt))
    frames, falls = [], 0
    start = d.qpos[:2].copy()
    for _ in range(int(args.seconds / env.dt)):
        obs, _, _, term, trunc = env.step(act(obs))
        renderer.update_scene(d, cam)
        frames.append(renderer.render())
        if term or trunc:
            falls += int(term)
            obs, _ = env.reset()
            start = d.qpos[:2].copy()
    imageio.mimsave(args.video, frames, fps=fps, macro_block_size=1)
    print(f"[INFO] wrote {args.video} ({len(frames)} frames, falls={falls}, last-episode travel={np.round(d.qpos[:2]-start, 3)} m)")


def interactive(env, act, obs):
    import mujoco
    import mujoco.viewer

    lock = threading.Lock()
    state = {"reset": False}

    def on_key(key):
        with lock:
            c = env.commands
            if key == KEY_UP:
                c[0] = min(c[0] + 0.03, 0.25)
            elif key == KEY_DOWN:
                c[0] = max(c[0] - 0.03, -0.12)
            elif key == KEY_LEFT:
                c[2] = min(c[2] + 0.15, 0.8)
            elif key == KEY_RIGHT:
                c[2] = max(c[2] - 0.15, -0.8)
            elif key == ord(","):
                c[1] = min(c[1] + 0.03, 0.1)
            elif key == ord("."):
                c[1] = max(c[1] - 0.03, -0.1)
            elif key == ord("K"):
                c[:] = 0.0
            elif key == KEY_BACKSPACE:
                state["reset"] = True
                return
            print(f"command vx={c[0]:+.2f} m/s  vy={c[1]:+.2f} m/s  yaw={c[2]:+.2f} rad/s")

    print(__doc__.split("Viewer keys")[1])
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.base_id
        viewer.cam.distance = 1.0
        while viewer.is_running():
            t0 = time.time()
            with lock:
                if state["reset"]:
                    obs, _ = env.reset()
                    state["reset"] = False
                obs, _, _, term, trunc = env.step(act(obs))
                if term or trunc:
                    obs, _ = env.reset()
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
