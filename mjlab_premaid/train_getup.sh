#!/usr/bin/env bash
# Train the get-up policy with mjlab on the GPU (Linux / WSL2). First run sets up ~/premaid_mjlab_venv.
#   ./mjlab_premaid/train_getup.sh                   # 4096 robots, 3000 iterations
#   ./mjlab_premaid/train_getup.sh --agent.resume True --agent.load-run <run dir name>
set -e
cd "$(dirname "$0")/.."
VENV="$HOME/premaid_mjlab_venv"
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install "mjlab==1.6.0" tensorboard
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no NVIDIA GPU visible - training will run on the CPU (slow)"
"$VENV/bin/python" -m mjlab_premaid.train Premaid-GetUp --env.scene.num-envs 4096 --log-root logs/mjlab "$@"
