import yaml
import wandb
import pyrallis
import torch
import numpy as np
from typing import Callable
from datetime import datetime
import dataclasses
from algorithms.sb3.ppo.ippo import IPPO
from algorithms.sb3.callbacks import MultiAgentCallback
from baselines.ippo.config import ExperimentConfig
from pygpudrive.env.config import EnvConfig, SceneConfig
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(env_config: EnvConfig, exp_config: ExperimentConfig, scene_config: SceneConfig, action_type: str = "discrete"):
    """Run PPO training with stable-baselines3."""
    # MAKE SB3-COMPATIBLE ENVIRONMENT
    env = SB3MultiAgentEnv(
        config=env_config,
        scene_config=scene_config,
        exp_config=exp_config,
        # Control up to all agents in the scene
        max_cont_agents=env_config.max_num_agents_in_scene,
        device=exp_config.device,
    )

    # SET MINIBATCH SIZE BASED ON ROLLOUT LENGTH
    exp_config.batch_size = (
        exp_config.num_worlds * exp_config.n_steps
    ) // exp_config.num_minibatches

    # INIT WANDB    
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"ppo_{datetime_}"
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    run = wandb.init(
        project=private_info['side_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id,
        group=f"{env_config.dynamics_model}_{action_type}",
        sync_tensorboard=exp_config.sync_tensorboard,
        tags=exp_config.tags,
        mode=exp_config.wandb_mode,
        config={**exp_config.__dict__, **env_config.__dict__},
    )

    # CALLBACK
    custom_callback = MultiAgentCallback(
        config=exp_config,
        wandb_run=run if run_id is not None else None,
    )

    # INITIALIZE IPPO
    model = IPPO(
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
        tensorboard_log=f"runs/{run_id}"
        if run_id is not None
        else None,  # Sync with wandb
        mlp_class=exp_config.mlp_class,
        policy=exp_config.policy,
        gamma=exp_config.gamma,
        gae_lambda=exp_config.gae_lambda,
        vf_coef=exp_config.vf_coef,
        clip_range=exp_config.clip_range,
        learning_rate=linear_schedule(exp_config.lr),
        ent_coef=exp_config.ent_coef,
        n_epochs=exp_config.n_epochs,
        env_config=env_config,
        exp_config=exp_config,
    )

    # LEARN
    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=custom_callback,
    )

    run.finish()
    env.close()


if __name__ == "__main__":

    exp_config = pyrallis.parse(config_class=ExperimentConfig)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 20), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 20), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-np.pi, np.pi, 20), decimals=3
        ),
    )

    scene_config = SceneConfig(
        path=exp_config.data_dir,
        num_scenes=exp_config.num_worlds,
        discipline=exp_config.selection_discipline,
        k_unique_scenes=exp_config.k_unique_scenes,
    )

    train(env_config, exp_config, scene_config, action_type="discrete")
