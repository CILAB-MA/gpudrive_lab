"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging
import os
import matplotlib.pyplot as plt

from gymnasium.spaces import Tuple, Discrete, MultiDiscrete

logging.getLogger(__name__)


def map_to_closest_discrete_value(grid, cont_actions):
    """
    Find the nearest value in the action grid for a given expert action.
    """
    # Calculate the absolute differences and find the indices of the minimum values
    abs_diff = torch.abs(grid.unsqueeze(0) - cont_actions.unsqueeze(-1))
    indx = torch.argmin(abs_diff, dim=-1)

    # Gather the closest values based on the indices
    closest_values = grid[indx]

    return closest_values, indx

def generate_state_action_pairs(
    env,
    make_video=False,
    render_index=[0, 1],
    save_path="./",
    debug_world_idx=None,
    debug_veh_idx=None,
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        make_video (bool): Whether to save a video of the expert trajectory.
        render_index (int): Index of the world to render (must be <= num_worlds).
        save_path (str): Path to save the video.
        debug_world_idx (int): Index of the world to debug.
        debug_veh_idx (int): Index of the vehicle to debug.

    Returns:
        expert_actions: Expert actions for the controlled agents. An action is a
            tuple with (acceleration, steering, heading).
        obs_tensor: Expert observations for the controlled agents.
    """
    frames = [[] for _ in range(render_index[1] - render_index[0])]

    logging.info(
        f"Generating expert actions and observations for {env.num_worlds} worlds \n"
    )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    expert_actions, debug_speeds, expert_positions = env.get_expert_actions(debug_world_idx, debug_veh_idx)
    raw_expert_action = expert_actions.clone()

    # Convert expert actions to the desired action space type
    dynamics_model = env.config.dynamics_model
    device = env.device
    
    if isinstance(env.action_space, Tuple):
        logging.info("Using continuous expert actions... \n")
    elif isinstance(env.action_space, Discrete) or isinstance(env.action_space, MultiDiscrete):
        logging.info(f"Converting expert actions into discretized format... \n")
        disc_expert_actions = torch.zeros_like(expert_actions)

        if dynamics_model == 'delta_local':
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.dx, cont_actions=expert_actions[:, :, :, 0]
            )
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.dy, cont_actions=expert_actions[:, :, :, 1]
            )
            disc_expert_actions[:, :, :, 2], _ = map_to_closest_discrete_value(
                grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2]
            )
        elif dynamics_model == 'classic' or dynamics_model == 'bicycle':
            # Acceleration
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
            )
            # Steering
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
            )
        else:
            raise NotImplementedError(f"Unsupported dynamics model: {dynamics_model}")
        expert_actions = disc_expert_actions
    else:
        raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")
    
    # Storage
    expert_observations_lst = []
    expert_actions_lst = []
    expert_next_obs_lst = []
    expert_dones_lst = []

    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device)
    alive_agent_mask = env.cont_agent_mask.clone().to(device)

    if debug_world_idx is not None and debug_veh_idx is not None:
        speeds = [obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1)]
        poss = [obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0)]

    for time_step in range(env.episode_len):
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :], use_indices=False)

        next_obs = env.get_obs()

        dones = env.get_dones().to(device)
        infos = env.get_infos()
        if debug_world_idx is not None and debug_veh_idx is not None:
            if dones[debug_world_idx, debug_veh_idx] == 0:
                speeds.append(next_obs[debug_world_idx, debug_veh_idx, 0].unsqueeze(-1))
                poss.append(next_obs[debug_world_idx, debug_veh_idx, 3:5].unsqueeze(0))

        # Unpack and store (obs, action, next_obs, dones) pairs for controlled agents
        expert_observations_lst.append(obs[~dead_agent_mask, :])
        expert_actions_lst.append(
            expert_actions[~dead_agent_mask][:, time_step, :]
        )
        expert_next_obs_lst.append(next_obs[~dead_agent_mask, :])
        expert_dones_lst.append(dones[~dead_agent_mask])

        # Update
        obs = next_obs
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        if make_video:
            for render in range(render_index[0], render_index[1]):
                frame = env.render(world_render_idx=render)
                frames[render].append(frame)
        if (dead_agent_mask == True).all():
            break
        
    controlled_agent_info = infos[alive_agent_mask]
    off_road = controlled_agent_info[:, 0]
    veh_collision = controlled_agent_info[:, 1]
    non_veh_collision = controlled_agent_info[:, 2]
    goal_achieved = controlled_agent_info[:, 3]        
    
    off_road_rate = off_road.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision.sum().float() / alive_agent_mask.sum().float()
    non_veh_coll_rate = non_veh_collision.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved.sum().float() / alive_agent_mask.sum().float()
    collision_rate = off_road_rate + veh_coll_rate + non_veh_coll_rate
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Non-vehCol {non_veh_coll_rate} Goal {goal_rate}')
    
    scene_collision = []
    for i in range(env.num_worlds):
        scene_info = infos[i,...][alive_agent_mask[i,...]]
        scene_off_road = scene_info[:, 0]
        scene_veh_collision = scene_info[:, 1]
        scene_non_veh_collision = scene_info[:, 2]
        scene_goal_achieved = scene_info[:, 3]
        
        scene_off_road_rate = scene_off_road.sum().float() / scene_off_road.size(0)
        scene_veh_collision_rate = scene_veh_collision.sum().float() / scene_veh_collision.size(0)
        scene_non_veh_collision_rate = scene_non_veh_collision.sum().float() / scene_non_veh_collision.size(0)
        scene_goal_rate = scene_goal_achieved.sum().float() / scene_goal_achieved.size(0)
        scene_collision_rate = scene_off_road_rate + scene_veh_collision_rate + scene_non_veh_collision_rate
        
        scene_collision.append(True if scene_collision_rate > 0.0 else False)
        print(f'World{i} : Offroad {scene_off_road_rate} VehCol {scene_veh_collision_rate} Non-vehCol {scene_non_veh_collision_rate} Goal {scene_goal_rate}')

    if make_video:
        if not os.path.exists(save_path):
            print(f"Error: {save_path} does not exist.")
        else:
            for render in range(render_index[0], render_index[1]):
                collision_status = "collided" if scene_collision[render] else "used"
                path = os.path.join(save_path, f"{type(env).__name__}_world_{render}_{collision_status}.mp4")
                imageio.mimwrite(path, np.array(frames[render]), fps=30)

    flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
    flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
    flat_expert_dones = torch.cat(expert_dones_lst, dim=0)
    
    if debug_world_idx is not None and debug_veh_idx is not None:
        '''for plotting '''
        debug_positions = expert_positions[debug_world_idx, debug_veh_idx]
        speeds = torch.cat(speeds)
        poss = torch.cat(poss, dim=0)
        
        if dynamics_model == 'delta_local':
            expert_dx = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
            expert_dy = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]
            expert_dyaw = raw_expert_action[debug_world_idx, debug_veh_idx, :, 2]
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            action_comb = (len(env.dx), len(env.dy), len(env.dyaw))
        elif dynamics_model == 'classic' or dynamics_model == 'bicycle':
            expert_accel = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
            expert_steer = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            action_comb = (len(env.accel_actions), len(env.steer_actions))
        elif dynamics_model == 'state':
            pass
        else:
            print('Error: Unsupported action features')
            
        # Speed plot
        axs[0, 0].plot(debug_speeds.cpu().numpy(), label='Expert Speeds', color='b')
        axs[0, 0].plot(speeds.cpu().numpy(), label='Simulation Speeds', color='r')
        axs[0, 0].set_title('Speeds Comparison')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Speed')
        axs[0, 0].legend()

        # Position plot
        axs[0, 1].plot(debug_positions[:, 0].cpu().numpy(), debug_positions[:, 1].cpu().numpy(), label='Expert Position',
                       color='b',
                       marker='o')
        axs[0, 1].plot(poss[:, 0].cpu().numpy(), poss[:, 1].cpu().numpy(), label='Environment Position', color='r', marker='x')
        axs[0, 1].set_title('Position Comparison with Order')
        axs[0, 1].set_xlabel('X Position')
        axs[0, 1].set_ylabel('Y Position')
        axs[0, 1].legend()
        
        if dynamics_model == 'delta_local':
            # dx plot
            axs[1, 0].plot(expert_dx.cpu().numpy(), label='Expert dx', color='b')
            axs[1, 0].plot(expert_actions[debug_world_idx, debug_veh_idx, :, 0].cpu().numpy(), label='Simulation dx',
                        color='r')
            axs[1, 0].set_title('dx Comparison')
            axs[1, 0].set_xlabel('Time Step')
            axs[1, 0].set_ylabel('dx')
            axs[1, 0].legend()

            # dy plot
            axs[1, 1].plot(expert_dy.cpu().numpy(), label='Expert dy', color='b')
            axs[1, 1].plot(expert_actions[debug_world_idx, debug_veh_idx, :, 1].cpu().numpy(), label='Simulation dy',
                        color='r')
            axs[1, 1].set_title('dy Comparison')
            axs[1, 1].set_xlabel('Time Step')
            axs[1, 1].set_ylabel('dy')
            axs[1, 1].legend()

            # dyaw plot
            axs[1, 2].plot(expert_dyaw.cpu().numpy(), label='Expert dyaw', color='b')
            axs[1, 2].plot(expert_actions[debug_world_idx, debug_veh_idx, :, 2].cpu().numpy(), label='Simulation dyaw',
                        color='r')
            axs[1, 2].set_title('dyaw Comparison')
            axs[1, 2].set_xlabel('Time Step')
            axs[1, 2].set_ylabel('dyaw')
            axs[1, 2].legend()
        elif dynamics_model == 'classic' or dynamics_model == 'bicycle':
            # Accels plot
            axs[1, 0].plot(expert_accel.cpu().numpy(), label='Expert Accels', color='b')
            axs[1, 0].plot(expert_actions[debug_world_idx, debug_veh_idx, :, 0].cpu().numpy(), label='Simulation Accels',
                        color='r')
            axs[1, 0].set_title('Accels Comparison')
            axs[1, 0].set_xlabel('Time Step')
            axs[1, 0].set_ylabel('Accels')
            axs[1, 0].legend()

            # Steers plot
            axs[1, 1].plot(expert_steer.cpu().numpy(), label='Expert Steers', color='b')
            axs[1, 1].plot(expert_actions[debug_world_idx, debug_veh_idx, :, 1].cpu().numpy(), label='Simulation Steers',
                        color='r')
            axs[1, 1].set_title('Steers Comparison')
            axs[1, 1].set_xlabel('Time Step')
            axs[1, 1].set_ylabel('Steers')
            axs[1, 1].legend()
        elif dynamics_model == 'state':
            pass
        else:
            pass

        plt.tight_layout()
        path = os.path.join(save_path, f"{type(env).__name__}_{action_comb}_W{debug_world_idx}_V{debug_veh_idx}.jpg")
        if not os.path.exists(save_path):
            print(f"Error: {save_path} does not exist.")
        else:
            plt.savefig(path, dpi=150)
        
    return (
        flat_expert_obs,
        flat_expert_actions,
        flat_next_expert_obs,
        flat_expert_dones,
        goal_rate,
        collision_rate,
    )


if __name__ == "__main__":
    from pygpudrive.registration import make
    from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    import argparse
    
    def parse_args():
        parser = argparse.ArgumentParser('Select the dynamics model that you use')
        parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
        parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
        args = parser.parse_args()
        return args
    
    args = parse_args()
    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = 5
    MAX_NUM_OBJECTS = 128

    # Initialize lists to store results
    num_actions = []
    goal_rates = []
    collision_rates = []

    # Set configurations
    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-3.0, 3.0, 20), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-3.0, 3.0, 20), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-1.0, 1.0, 20), decimals=3
        ),
    )
    
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "max_cont_agents": MAX_NUM_OBJECTS,
        "device": args.device,
        "render_config": render_config,
        "num_stack": 3,
    }

    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_id=ActionSpace.DISCRETE, kwargs=kwargs)

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
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, NUM_WORLDS],  #start_idx, end_idx
        save_path="./",
        debug_world_idx=None,
        debug_veh_idx=None,
    )
    env.close()
    del env
    del env_config
