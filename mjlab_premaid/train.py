"""Train Premaid tasks with mjlab (GPU).  python -m mjlab_premaid.train Premaid-GetUp --env.scene.num-envs 4096"""
import mjlab_premaid  # noqa: F401  (registers the tasks)
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
