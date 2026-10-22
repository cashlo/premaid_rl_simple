"""Turn an mjlab (rsl_rl) checkpoint into the TorchScript file the deploy script / CPU player load.

    python mjlab_premaid/export.py logs/rsl_rl/premaid_getup/<run>/model_1500.pt  [-o getup_policy.pt]

Output: obs(86) -> action mean(25), observation normalizer baked in. Then e.g.
    python mujoco_sim/deploy/deploy_policy.py --mode getup --getup_checkpoint getup_policy.pt --sim --render
"""
import argparse
import os

import torch
from torch import nn


class ExportedActor(nn.Module):
    def __init__(self, sd: dict, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].reshape(1, -1).clone())
        self.register_buffer("std", sd["obs_normalizer._std"].reshape(1, -1).clone() + eps)
        ids = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
        layers = []
        for i, j in enumerate(ids):
            w, b = sd[f"mlp.{j}.weight"], sd[f"mlp.{j}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if i < len(ids) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / self.std)


def export(ckpt: str, out: str | None = None) -> str:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    net = torch.jit.script(ExportedActor(sd).eval())
    out = out or os.path.join(os.path.dirname(ckpt), "policy_deploy.pt")
    net.save(out)
    print(f"saved {out}  (obs {sd['mlp.0.weight'].shape[1]} -> act {net(torch.zeros(1, sd['mlp.0.weight'].shape[1])).shape[1]})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    export(a.checkpoint, a.out)
