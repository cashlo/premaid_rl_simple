"""Task registry: which config/env class and run folder each task uses."""

from getup_env import GetUpCfg, PremaidGetUpEnv
from premaid_env import PremaidWalkEnv, WalkCfg

TASKS = {
    "walk": dict(cfg=WalkCfg, env=PremaidWalkEnv, run_dir="runs/premaid_mujoco"),
    "getup": dict(cfg=GetUpCfg, env=PremaidGetUpEnv, run_dir="runs/premaid_getup_v2"),  # v2: fresh start after the init/filter fixes
}


def make_env(cfg, seed=0):
    cls = PremaidGetUpEnv if isinstance(cfg, GetUpCfg) else PremaidWalkEnv
    return cls(cfg, seed=seed)
