import numpy as np
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stable_baselines3.common import policies
import os, sys, torch, argparse
sys.path.append(os.getcwd())
# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_delta_torch import GPUDriveTorchEnv

from algorithms.il.data_generation_delta import generate_state_action_pairs
from baselines.il.config import BehavCloningConfig

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-d', type=str, default='delta')
    args = parser.parse_args()
    return args



class CustomFeedForwardPolicy(policies.ActorCriticPolicy):
    """A feed forward policy network with a number of hidden units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=bc_config.net_arch)


def train_bc(
    env_config,
    scene_config,
    render_config,
    bc_config,
):
    rng = np.random.default_rng()

    # Make env
    # env = GPUDriveTorchEnv(
    #     config=env_config,
    #     scene_config=scene_config,
    #     max_cont_agents=bc_config.max_cont_agents,
    #     render_config=render_config,
    #     device=bc_config.device,
    # )
    # if bc_config.wandb_mode == 'online':
    #     with open("private.yaml") as f:
    #         private_info = yaml.load(f, Loader=yaml.FullLoader)
    #     wandb.login(key=private_info["wandb_key"])
    #     wandb.init(project=private_info["project"], entity=private_info["entity"],
    #                name='bc_test')

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=bc_config.max_cont_agents,  # Number of agents to control
        device="cpu",
        render_config=render_config,
        useDeltaModel=True,
    )
    # Make custom policy
    policy = CustomFeedForwardPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )

    # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
        goal_rate,
        collision_rate
    ) = generate_state_action_pairs(
        env=env,
        device="cpu",
        discretize_actions=True,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, 0],  # start_idx, end_idx
        debug_world_idx=None,
        debug_veh_idx=None,
        save_path="use_discr_actions_fix",
        num_action=300*100*100
    )

    # Convert to dataset of imitation "transitions"
    transitions = Transitions(
        obs=expert_obs.cpu().numpy(),
        acts=expert_actions.cpu().numpy(),
        infos=np.zeros_like(expert_dones.cpu()),  # Dummy
        next_obs=next_expert_obs.cpu().numpy(),
        dones=expert_dones.cpu().numpy().astype(bool),
    )

    # Define trainer
    bc_trainer = bc.BC(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        device=torch.device("cpu"),
    )

    # Train
    bc_trainer.train(
        n_epochs=bc_config.epochs,
        log_interval=bc_config.log_interval,
    )

    # Save policy
    if bc_config.save_model:
        bc_trainer.policy.save(
            path=f"{bc_config.model_path}/{bc_config.model_name}.pt"
        )


if __name__ == "__main__":
    import argparse
    args = parse_args()
    NUM_WORLDS = 1000

    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-2.0, 2.0, 100), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-2.0, 2.0, 100), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-1.0, 1.0, 300), decimals=3
        ),
    )
    render_config = RenderConfig()
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    bc_config = BehavCloningConfig()

    train_bc(
        env_config=env_config,
        render_config=render_config,
        scene_config=scene_config,
        bc_config=bc_config,
    )
