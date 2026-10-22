"""Watch a policy:  python -m mjlab_premaid.play Premaid-GetUp --checkpoint-file logs/rsl_rl/premaid_getup/<run>/model_X.pt"""
import mjlab_premaid  # noqa: F401
from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
