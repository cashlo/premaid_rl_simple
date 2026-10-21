"""Train Premaid AI to walk in MuJoCo with skrl PPO (asymmetric actor-critic).

    cd mujoco_sim
    python train_mujoco.py                          # fresh run, all CPU cores
    python train_mujoco.py --num_envs 512 --workers 15
    python train_mujoco.py --resume latest          # continue from newest checkpoint
    python train_mujoco.py --task getup             # get-up-from-lying task (runs/premaid_getup)

Logs + checkpoints go to runs/premaid_mujoco/<time>_PPO/ (TensorBoard: `tensorboard --logdir runs`).
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
import torch._dynamo  # noqa: F401  (import before mujoco: avoids a segfault with MUJOCO_GL=osmesa)
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer

from tasks import TASKS
from vec_env import SkrlPremaidWrapper, SubprocVecPremaid

RUN_DIR = os.path.join("runs", "premaid_mujoco")


def mlp(n_in, n_out, hidden=(512, 256, 128)):
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class Policy(GaussianMixin, Model):
    """Actor: sees only what the real robot can sense (observations)."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5.0, max_log_std=1.0)
        self.net = mlp(self.num_observations, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    """Critic: sees privileged simulator state (base velocity, contacts, friction...)."""

    def __init__(self, state_space, device):
        Model.__init__(self, state_space=state_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = mlp(self.num_states, 1)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


def make_agent(env, device, rollouts=24, run_dir=RUN_DIR):
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.state_space, device),
    }
    cfg = PPO_CFG(
        rollouts=rollouts,
        learning_epochs=5,
        mini_batches=4,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        learning_rate_scheduler=KLAdaptiveLR,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": env.observation_space, "device": device},
        state_preprocessor=RunningStandardScaler,
        state_preprocessor_kwargs={"size": env.state_space, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.002,
        value_loss_scale=1.0,
        time_limit_bootstrap=True,
    )
    cfg.experiment.directory = run_dir
    cfg.experiment.write_interval = 50
    cfg.experiment.checkpoint_interval = 1000
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)
    agent = PPO(
        models=models, memory=memory, cfg=cfg, observation_space=env.observation_space,
        state_space=env.state_space, action_space=env.action_space, device=device,
    )
    return agent


def find_checkpoint(spec: str, run_dir: str = RUN_DIR) -> str:
    if spec != "latest":
        return spec
    cands = glob.glob(os.path.join(run_dir, "*", "checkpoints", "*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {run_dir}/")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=None, help="CPU processes (default: cores - 1)")
    ap.add_argument("--timesteps", type=int, default=200_000, help="vectorized steps (x num_envs = samples)")
    ap.add_argument("--rollouts", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="checkpoint path or 'latest'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="walk", choices=list(TASKS))
    ap.add_argument("--no_randomize", action="store_true", help="disable domain randomization/pushes")
    ap.add_argument("--viewer", action="store_true", help="open a live MuJoCo window showing robot #0 while training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    task = TASKS[args.task]
    cfg = task["cfg"](randomize=not args.no_randomize)
    vec = SubprocVecPremaid(args.num_envs, args.workers, seed=args.seed, cfg=cfg, viewer=args.viewer)
    env = SkrlPremaidWrapper(vec, device=args.device)
    gpu = f" ({torch.cuda.get_device_name(0)})" if str(args.device).startswith("cuda") else ""
    print(f"[INFO] {env.num_envs} robots on {len(vec.procs)} CPU workers, learner on {args.device}{gpu}")

    print(f"[INFO] task: {args.task}")
    agent = make_agent(env, args.device, args.rollouts, run_dir=task["run_dir"])
    if args.resume:
        path = find_checkpoint(args.resume, task["run_dir"])
        print(f"[INFO] resuming from {path}")
        agent.load(path)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": True}, env=env, agents=agent)
    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
