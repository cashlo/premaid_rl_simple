"""Train Premaid AI to walk in Isaac Lab (fixed version of train_direct.py).

Run from the folder that has premaidai.usd + configuration/ (your IsaacLab folder):

    isaaclab.bat -p train_walk.py --headless
    isaaclab.bat -p train_walk.py --num_envs 64              # watch it in the viewport
    isaaclab.bat -p train_walk.py --headless --resume latest

Checkpoints + TensorBoard logs: runs/premaid_walk/<time>_PPO/
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Premaid AI walking (Isaac Lab + skrl PPO)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--timesteps", type=int, default=30_000, help="policy steps per robot (x num_envs = samples)")
parser.add_argument("--resume", default=None, help="checkpoint path or 'latest'")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402

from skrl_compat import find_checkpoint, make_ppo  # noqa: E402
from walk_env import NUM_ACTOR_OBS, PremaidWalkEnv, PremaidWalkEnvCfg  # noqa: E402

RUN_DIR = os.path.join("runs", "premaid_walk")


def main():
    torch.manual_seed(args_cli.seed)
    cfg = PremaidWalkEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    if args_cli.device:
        cfg.sim.device = args_cli.device
    env = SkrlVecEnvWrapper(PremaidWalkEnv(cfg))

    agent = make_ppo(env, NUM_ACTOR_OBS, RUN_DIR)
    if args_cli.resume:
        path = find_checkpoint(args_cli.resume, RUN_DIR)
        print(f"[INFO] resuming from {path}")
        agent.load(path)

    trainer = SequentialTrainer(
        cfg={"timesteps": args_cli.timesteps, "headless": True, "environment_info": "log"}, env=env, agents=agent
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
