"""Small shim so the walking scripts run on skrl 1.x and 2.x."""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
import skrl
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler

SKRL2 = int(skrl.__version__.split(".")[0]) >= 2

try:  # skrl 1.x name, then 2.x name
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except ImportError:  # pragma: no cover
    from skrl.resources.schedulers.torch import KLAdaptiveRL as KLAdaptiveLR

if SKRL2:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
else:
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG


def _obs(inputs):
    return inputs["observations"] if "observations" in inputs else inputs["states"]


def mlp(n_in, n_out, hidden=(512, 256, 128)):
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


class Policy(GaussianMixin, Model):
    """Actor. Only reads the first `num_actor_obs` entries (what the real robot can sense)."""

    def __init__(self, observation_space, action_space, device, num_actor_obs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5.0, max_log_std=1.0)
        self.num_actor_obs = num_actor_obs
        self.net = mlp(num_actor_obs, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -1.0))

    def compute(self, inputs, role=""):
        mean = self.net(_obs(inputs)[:, : self.num_actor_obs])
        if SKRL2:
            return mean, {"log_std": self.log_std_parameter}
        return mean, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Critic. Reads everything, including the privileged simulator terms."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = mlp(self.num_observations, 1)

    def compute(self, inputs, role=""):
        return self.net(_obs(inputs)), {}


def make_ppo(env, num_actor_obs, directory, rollouts=24, write_interval=50, checkpoint_interval=500):
    dev = env.device
    models = {
        "policy": Policy(env.observation_space, env.action_space, dev, num_actor_obs),
        "value": Value(env.observation_space, env.action_space, dev),
    }
    common = dict(
        rollouts=rollouts,
        learning_epochs=5,
        mini_batches=4,
        discount_factor=0.99,
        learning_rate=1e-3,
        learning_rate_scheduler=KLAdaptiveLR,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": dev},
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.002,
        value_loss_scale=1.0,
        time_limit_bootstrap=True,
    )
    obs_scaler = {"size": env.observation_space, "device": dev}
    if SKRL2:
        cfg = PPO_CFG(
            **common, gae_lambda=0.95,
            observation_preprocessor=RunningStandardScaler, observation_preprocessor_kwargs=obs_scaler,
        )
        cfg.experiment.directory = directory
        cfg.experiment.write_interval = write_interval
        cfg.experiment.checkpoint_interval = checkpoint_interval
    else:
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg.update(common)
        cfg["lambda"] = 0.95
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = obs_scaler
        cfg["experiment"]["directory"] = directory
        cfg["experiment"]["write_interval"] = write_interval
        cfg["experiment"]["checkpoint_interval"] = checkpoint_interval
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=dev)
    return PPO(
        models=models, memory=memory, cfg=cfg,
        observation_space=env.observation_space, action_space=env.action_space, device=dev,
    )


def deterministic_action(agent, obs):
    """Mean action of the policy for raw env observations (used for playback/deployment)."""
    pre = agent._observation_preprocessor if SKRL2 else agent._state_preprocessor
    x = pre(obs)
    with torch.no_grad():
        return agent.policy.net(x[:, : agent.policy.num_actor_obs])


def set_eval(agent):
    if SKRL2:
        agent.enable_training_mode(False, apply_to_models=True)
    else:
        agent.set_running_mode("eval")


def find_checkpoint(spec, run_dir):
    if spec != "latest":
        return spec
    cands = glob.glob(os.path.join(run_dir, "*", "checkpoints", "*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {run_dir}/")
    return max(cands, key=os.path.getmtime)
