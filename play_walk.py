"""Watch / drive the Isaac Lab walking policy with the keyboard.

    isaaclab.bat -p play_walk.py                       # newest checkpoint in runs/premaid_walk
    isaaclab.bat -p play_walk.py --checkpoint runs/premaid_walk/.../agent_6000.pt --num_envs 16
    isaaclab.bat -p play_walk.py --export premaid_walk_policy.pt   # TorchScript for the real robot

Keys (click the viewport first): Up/Down forward/back, Left/Right strafe, Z/X turn.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Premaid AI walking playback")
parser.add_argument("--checkpoint", default="latest")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--export", default=None, help="save a TorchScript policy (86 obs -> 25 actions) and exit")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

from skrl_compat import SKRL2, deterministic_action, find_checkpoint, make_ppo, set_eval  # noqa: E402
from walk_env import NUM_ACTOR_OBS, PremaidWalkEnv, PremaidWalkEnvCfg  # noqa: E402


class ExportedPolicy(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        pre = agent._observation_preprocessor if SKRL2 else agent._state_preprocessor
        self.register_buffer("mean", pre.running_mean[:NUM_ACTOR_OBS].clone().float().cpu())
        self.register_buffer("std", pre.running_variance[:NUM_ACTOR_OBS].clone().float().cpu().sqrt())
        self.eps, self.clip = float(pre.epsilon), float(pre.clip_threshold)
        self.net = agent.policy.net.cpu()

    def forward(self, obs):
        return self.net(torch.clamp((obs - self.mean) / (self.std + self.eps), -self.clip, self.clip))


def main():
    cfg = PremaidWalkEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.obs_noise = False
    cfg.push_interval_s = (1e9, 1e9)  # no pushes while you drive
    env = SkrlVecEnvWrapper(PremaidWalkEnv(cfg))
    raw = env.unwrapped
    raw.manual_commands = True
    raw.commands[:] = 0.0

    agent = make_ppo(env, NUM_ACTOR_OBS, "runs/premaid_walk_play")
    path = find_checkpoint(args_cli.checkpoint, "runs/premaid_walk")
    print(f"[INFO] loading {path}")
    agent.load(path)
    set_eval(agent)

    if args_cli.export:
        torch.jit.script(ExportedPolicy(agent)).save(args_cli.export)
        print(f"[INFO] exported {args_cli.export}: input = first {NUM_ACTOR_OBS} obs of walk_env, output = 25 actions "
              f"in Isaac joint order {raw.robot.joint_names}")
        env.close()
        return

    keyboard = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=0.15, v_y_sensitivity=0.08, omega_z_sensitivity=0.6))
    print(__doc__.split("Keys")[1])
    obs, _ = env.reset()
    while simulation_app.is_running():
        cmd = torch.as_tensor(keyboard.advance(), dtype=torch.float32, device=raw.device)
        raw.commands[:] = cmd
        actions = deterministic_action(agent, obs)
        obs, _, _, _, _ = env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
