"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
import os, sys, torch
sys.path.append(os.getcwd())
import argparse
import wandb, yaml
from datetime import datetime
from typing import Callable
from algorithms.sb3.ppo.ippo import IPPO
from stable_baselines3.common.callbacks import CallbackList
from algorithms.sb3.callbacks import MultiAgentCallback, StdoutCallback
from baselines.ippo.config import ExperimentConfig
from pygpudrive.env.config import EnvConfig, SceneConfig, ActionSpace
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'])
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num-worlds', '-w', type=int, default=10)
    parser.add_argument('--num-scenes', '-s', type=int, default=1)
    parser.add_argument('--num-stack', type=int, default=5)

    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/app/model')
    parser.add_argument('--model-name', '-m', type=str, default='late_fusion', choices=['bc', 'late_fusion', 'attention', 'wayformer'])
    parser.add_argument('--loss-name', '-l', type=str, default='gmm', choices=['l1', 'mse', 'twohot', 'nll', 'gmm'])
    parser.add_argument('--action-scale', '-as', type=int, default=1)

    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--weights', '-we', type=str, default='from_scratch', choices=["from_scratch", "from_bc"])
    parser.add_argument('--wandb-path', '-wp', type=str, default=None)
    args = parser.parse_args()

    return args


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


if __name__ == "__main__":
    args = parse_args()

    # Load Model
    method = f"{args.model_name}_{args.loss_name}_{args.exp_name}"
    bc_policy = torch.load(f"{args.model_path}/{method}.pth")

    # Make configs
    exp_config = ExperimentConfig(
        num_worlds=args.num_worlds,
        k_unique_scenes=args.num_scenes,
        resample_scenarios=False,
        device=args.device,
    )
    exp_config.mlp_class = bc_policy.__class__
    for key in dir(bc_policy.net_config):
        attr = getattr(bc_policy.net_config, key)
        if not callable(attr) and not key.startswith("__"):
            setattr(exp_config, key, attr)
    env_config = EnvConfig(
        dynamics_model='delta_local',
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to(args.device),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to(args.device),
        dyaw=torch.round(
            torch.linspace(-np.pi, np.pi, 100), decimals=3
        ).to(args.device),
    )
    exp_config.batch_size = (exp_config.num_worlds * exp_config.n_steps) // exp_config.num_minibatches  # SET MINIBATCH SIZE BASED ON ROLLOUT LENGTH
    scene_config = SceneConfig(
        path=exp_config.data_dir,
        num_scenes=exp_config.num_worlds,
        discipline=exp_config.selection_discipline,
        k_unique_scenes=exp_config.k_unique_scenes,
    )

    # Initialize environment
    env = SB3MultiAgentEnv(
        config=env_config,
        scene_config=scene_config,
        exp_config=exp_config,
        # Control up to all agents in the scene
        max_cont_agents=env_config.max_num_agents_in_scene,
        device=exp_config.device,
        action_space=ActionSpace.CONTINUOUS if args.action_type == 'continuous' else None,
        num_stack=args.num_stack,
    )

    # INITIALIZE IPPO
    model = IPPO(
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        # verbose=exp_config.verbose,
        verbose=1,
        device=exp_config.device,
        mlp_class=exp_config.mlp_class,
        mlp_config={'loss': args.loss_name},
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
    if args.weights == "from_bc":
        model.policy.mlp_extractor.load_state_dict(bc_policy.state_dict())

    # CALLBACKS
    callbacks = [StdoutCallback(config=exp_config)]

    # INIT WANDB
    if args.wandb_path:
        with open(args.wandb_path) as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        run = wandb.init(
            project=private_info['project'],
            entity=private_info['entity'],
            name=method + "_" + datetime.now().strftime("%m%d%H%M"),
            group=args.weights,
            config={**exp_config.__dict__, **env_config.__dict__},
            tags=[args.model_name, args.loss_name, args.exp_name, str(args.num_scenes)]
        )
        wandb.config.update({
            'num_worlds': args.num_worlds,
            'num_scenes': args.num_scenes,
            'num_stack': args.num_stack,
            'num_vehicle': 128,
        })
        callbacks.append(MultiAgentCallback(config=exp_config, wandb_run=run))

    # LEARN
    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=CallbackList(callbacks),
    )
    env.close()
