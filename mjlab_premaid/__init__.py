"""Premaid AI tasks for mjlab. Importing this package registers them."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task

from .getup import premaid_getup_env_cfg

GETUP_TASK = "Premaid-GetUp"


def getup_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="premaid_getup",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=3000,
        upload_model=False,
    )


register_mjlab_task(
    task_id=GETUP_TASK,
    env_cfg=premaid_getup_env_cfg(),
    play_env_cfg=premaid_getup_env_cfg(play=True),
    rl_cfg=getup_rl_cfg(),
)
