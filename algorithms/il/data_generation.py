"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging
import os
import itertools
import matplotlib.pyplot as plt
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

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
        use_action_indices=False,
        make_video=False,
        render_index=[0, 1],
        save_path="./",
        debug_world_idx=None,
        debug_veh_idx=None,
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        use_action_indices (bool): Whether to return action indices instead of action values.
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
    print(f'EXPERT POSITIONS {expert_positions[0,0]}')
    num_scene, num_vehicle, timestep, _ = expert_positions.shape
    positions_expanded_1 = expert_positions.unsqueeze(2)  # torch.Size([5, 128, 1, 91, 2])
    positions_expanded_2 = expert_positions.unsqueeze(1)  # torch.Size([5, 1, 128, 91, 2])

    distances = torch.norm(positions_expanded_1 - positions_expanded_2, dim=-1)  # torch.Size([5, 128, 128, 91])

    mask = torch.eye(num_vehicle, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)  # torch.Size([1, 128, 128, 1])
    distances_masked = distances.masked_fill(mask, float('inf'))
    sorted_indices = distances_masked.argsort(dim=2)  # torch.Size([5, 128, 128, 91])

    other_actions = expert_actions.unsqueeze(2).expand(num_scene, num_vehicle, num_vehicle, timestep,
                                                       3)  # torch.Size([5, 128, 128, 91, 3])

    sorted_actions = torch.gather(other_actions, 2, sorted_indices.unsqueeze(-1).expand(-1, -1, -1, -1,
                                                                                        3))  # torch.Size([5, 128, 128, 91, 3])
    sorted_actions_without_self = sorted_actions[:, :, 1:, :, :]  # torch.Size([5, 128, 127, 91, 3])
    sorted_actions_final = sorted_actions_without_self.permute(0, 1, 3, 2, 4).reshape(num_scene, num_vehicle,
                                                                                      timestep,
                                                                                      (num_vehicle - 1) * 3)
    print(f'SORTED action {sorted_actions_final}')
    debug_positions = expert_positions[debug_world_idx, debug_veh_idx]
    raw_expert_action = expert_actions.clone()
    expert_dx, expert_dy, expert_dyaw = None, None, None
    expert_accel, expert_steer = None, None
    
    if debug_world_idx is not None and debug_veh_idx is not None:
        if env.action_features == 'delta_local':
            expert_dx = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
            expert_dy = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]
            expert_dyaw = raw_expert_action[debug_world_idx, debug_veh_idx, :, 2]
        else:
            expert_accel = raw_expert_action[debug_world_idx, debug_veh_idx, :, 0]
            expert_steer = raw_expert_action[debug_world_idx, debug_veh_idx, :, 1]

    # Convert expert actions to the desired action space type
    action_type = env.action_type
    device = env.device
    
    if action_type == 'continuous':
        logging.info("Using continuous expert actions... \n")
    else:
        logging.info(f"Converting expert actions into {action_type} format... \n")
        disc_expert_actions = torch.zeros_like(expert_actions)

        if env.action_features == 'delta_local':
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.dx, cont_actions=expert_actions[:, :, :, 0]
            )
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.dy, cont_actions=expert_actions[:, :, :, 1]
            )
            disc_expert_actions[:, :, :, 2], _ = map_to_closest_discrete_value(
                grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2]
            )
        else:
            # Acceleration
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
            )
            # Steering
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
            )

        if use_action_indices:  
            logging.info("Mapping expert actions to joint action index... \n")
            
            action_indices = 1 if action_type == 'discrete' else 3
            expert_action_indices = torch.zeros(
                expert_actions.shape[0],
                expert_actions.shape[1],
                expert_actions.shape[2],
                action_indices,
                dtype=torch.int32,
                ).to(device)
            
            for world_idx in range(disc_expert_actions.shape[0]):
                for agent_idx in range(disc_expert_actions.shape[1]):
                    for time_idx in range(disc_expert_actions.shape[2]):
                        action_val_tuple = tuple(
                            round(x, 3)
                            for x in disc_expert_actions[
                                     world_idx, agent_idx, time_idx, :
                                     ].tolist()
                        )
                        if not env.action_features == 'delta_local':
                            action_val_tuple = (action_val_tuple[0], action_val_tuple[1], 0.0)

                        action_idx = env.values_to_action_key.get(
                            action_val_tuple
                        )
                        expert_action_indices[
                            world_idx, agent_idx, time_idx
                        ] = torch.tensor(action_idx)

            expert_actions = expert_action_indices
        else:
            expert_actions = disc_expert_actions

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
        env.step_dynamics(expert_actions[:, :, time_step, :], use_action_indices)

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

    if debug_world_idx is not None:
        speeds = torch.cat(speeds)
        poss = torch.cat(poss, dim=0)

    if make_video:
        if not os.path.exists(save_path):
            print(f"Error: {save_path} does not exist.")
        else:
            for render in range(render_index[0], render_index[1]):
                collision_status = "collided" if scene_collision[render] else "used"
                path = os.path.join(save_path, f"world_{render}_{collision_status}_{action_type}.mp4")
                imageio.mimwrite(path, np.array(frames[render]), fps=30)

    flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
    flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
    flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
    flat_expert_dones = torch.cat(expert_dones_lst, dim=0)
    
    if debug_world_idx is not None:
        '''for plotting '''
        if env.action_features == 'delta_local':
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            action_comb = (len(env.dx), len(env.dy), len(env.dyaw))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            action_comb = (len(env.accel_actions), len(env.steer_actions))
            
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
        
        if not use_action_indices or action_type == 'continuous':
            if env.action_features == 'delta_local':
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
            else:
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
                axs[1, 1].plot(disc_expert_actions[debug_world_idx, debug_veh_idx, :, 1].cpu().numpy(), label='Simulation Steers',
                            color='r')
                axs[1, 1].set_title('Steers Comparison')
                axs[1, 1].set_xlabel('Time Step')
                axs[1, 1].set_ylabel('Steers')
                axs[1, 1].legend()

        plt.tight_layout()
        path = os.path.join(save_path, f"{action_type}_Action_{action_comb}_W{debug_world_idx}_V{debug_veh_idx}.jpg")
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
        collision_rate
    )


if __name__ == "__main__":
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser('Select the dynamics model that you use')
        parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
        parser.add_argument('--action-type', '-at', type=str, default='multi_discrete', choices=['discrete', 'multi_discrete', 'continuous'],)
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

    # Set the environment and render configurations
    # Action space (joint discrete)
    num_dx_values = reversed(range(300, 301, 1))
    num_dy_values = reversed(range(300, 301, 1))
    num_dyaw_values = reversed(range(300, 301, 1))

    combinations = itertools.product(num_dx_values, num_dy_values, num_dyaw_values)
    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", NUM_WORLDS, start_idx=5)
    for combi in combinations:
        num_dx, num_dy, num_dyaw = combi
        env_config = EnvConfig(
            dynamics_model=args.dynamics_model,
            steer_actions=torch.round(
                torch.linspace(-0.3, 0.3, 7), decimals=3
            ),
            accel_actions=torch.round(
                torch.linspace(-6.0, 6.0, 7), decimals=3
            ),
            dx=torch.round(
                torch.linspace(-3.0, 3.0, num_dx), decimals=3
            ),
            dy=torch.round(
                torch.linspace(-3.0, 3.0, num_dy), decimals=3
            ),
            dyaw=torch.round(
                torch.linspace(-1.0, 1.0, num_dyaw), decimals=3
            ),
        )

        env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
            action_type=args.action_type,
            device=args.device,
            render_config=render_config,
            num_stack=3
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
            use_action_indices=False,  # Map action values to joint action index
            make_video=True,  # Record the trajectories as sanity check
            render_index=[0, NUM_WORLDS],  #start_idx, end_idx
            save_path="./",
            debug_world_idx=0,
            debug_veh_idx=0,
        )
        env.close()
        del env
        del env_config

        # Store the results
        num_action = num_dx * num_dy * num_dyaw
        num_actions.append(num_action)
        goal_rates.append(goal_rate.cpu().numpy())
        collision_rates.append(collision_rate.cpu().numpy())
        print(f'\nCollision rate {collision_rate} Goal RATE {goal_rate}')

    # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(num_actions, goal_rates, label='Goal Rate', marker='o')
    # plt.plot(num_actions, collision_rates, label='Collision Rate', marker='x')
    # plt.xlabel('Number of Actions')
    # plt.ylabel('Rate')
    # plt.title('Goal Rate and Collision Rate vs. Number of Actions')
    # plt.legend()
    # plt.savefig('Trade-off.jpg',dpi=300)
    # Uncommment to save the expert actions and observations
    # torch.save(expert_actions, "expert_actions.pt")
    # torch.save(expert_obs, "expert_obs.pt")
