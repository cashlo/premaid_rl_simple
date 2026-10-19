"""Run many PremaidWalkEnv instances across CPU cores and expose them to skrl.

Each worker process owns `num_envs / num_workers` robots and steps them
sequentially; workers run in parallel. Environments auto-reset (Isaac-Lab
style), so the observation returned for a finished robot is already the first
observation of its next episode.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import gymnasium
import numpy as np
import torch
from skrl.envs.wrappers.torch.base import Wrapper

from premaid_env import NUM_ACTIONS, NUM_OBS, NUM_STATES, PremaidWalkEnv, WalkCfg


def _worker(remote, parent_remote, n, seed, cfg):
    parent_remote.close()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    envs = [PremaidWalkEnv(cfg, seed=seed + i) for i in range(n)]
    obs = np.zeros((n, NUM_OBS), np.float32)
    st = np.zeros((n, NUM_STATES), np.float32)
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                for i, e in enumerate(envs):
                    obs[i], st[i] = e.reset()
                remote.send((obs.copy(), st.copy()))
            elif cmd == "step":
                rew = np.zeros(n, np.float32)
                term = np.zeros(n, bool)
                trunc = np.zeros(n, bool)
                summaries = []
                for i, e in enumerate(envs):
                    o, s, rew[i], term[i], trunc[i] = e.step(payload[i])
                    if term[i] or trunc[i]:
                        summaries.append(e.episode_summary())
                        o, s = e.reset()
                    obs[i], st[i] = o, s
                remote.send((obs.copy(), st.copy(), rew, term, trunc, summaries))
            elif cmd == "close":
                break
    finally:
        remote.close()


class SubprocVecPremaid:
    def __init__(self, num_envs: int, num_workers: int | None = None, seed: int = 0, cfg: WalkCfg | None = None):
        cfg = cfg or WalkCfg()
        num_workers = num_workers or max(1, (os.cpu_count() or 2) - 1)
        num_workers = min(num_workers, num_envs)
        counts = [num_envs // num_workers + (1 if i < num_envs % num_workers else 0) for i in range(num_workers)]
        self.slices = np.cumsum([0] + counts)
        self.num_envs = num_envs
        ctx = mp.get_context("spawn" if os.name == "nt" else "forkserver")
        self.remotes, self.procs = [], []
        for w, n in enumerate(counts):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, parent, n, seed + 10_000 * w, cfg), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        return np.concatenate([x[0] for x in res]), np.concatenate([x[1] for x in res])

    def step(self, actions: np.ndarray):
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[self.slices[w]: self.slices[w + 1]]))
        res = [r.recv() for r in self.remotes]
        cat = lambda k: np.concatenate([x[k] for x in res])  # noqa: E731
        summaries = [s for x in res for s in x[5]]
        return cat(0), cat(1), cat(2), cat(3), cat(4), summaries

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=2)


class SkrlPremaidWrapper(Wrapper):
    """skrl Wrapper: torch tensors in/out, observations for the actor, states for the critic."""

    def __init__(self, vec: SubprocVecPremaid, device: str | torch.device = "cpu"):
        self._vec = vec
        super().__init__(vec)
        self._device = torch.device(device)
        self._obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (NUM_OBS,), np.float32)
        self._state_space = gymnasium.spaces.Box(-np.inf, np.inf, (NUM_STATES,), np.float32)
        self._act_space = gymnasium.spaces.Box(-1.0, 1.0, (NUM_ACTIONS,), np.float32)
        self._states = None

    # spaces / sizes
    @property
    def observation_space(self):
        return self._obs_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._act_space

    @property
    def num_envs(self) -> int:
        return self._vec.num_envs

    @property
    def num_agents(self) -> int:
        return 1

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self._device)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, st = self._vec.reset()
        self._states = self._t(st)
        return self._t(obs), {}

    def step(self, actions: torch.Tensor):
        obs, st, rew, term, trunc, summaries = self._vec.step(actions.detach().cpu().numpy())
        self._states = self._t(st)
        infos: dict[str, Any] = {}
        if summaries:
            keys = summaries[0].keys()
            infos["episode"] = {k: torch.tensor(float(np.mean([s[k] for s in summaries]))) for k in keys}
        return (
            self._t(obs),
            self._t(rew).view(-1, 1),
            self._t(term, torch.bool).view(-1, 1),
            self._t(trunc, torch.bool).view(-1, 1),
            infos,
        )

    def state(self) -> torch.Tensor | None:
        return self._states

    def render(self, *args, **kwargs):
        return None

    def close(self) -> None:
        self._vec.close()
