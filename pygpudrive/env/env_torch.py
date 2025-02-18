"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Tuple, Discrete, MultiDiscrete
import numpy as np
import torch
import imageio
from itertools import product

import gpudrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv
from pygpudrive.env import constants

class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        num_stack=1,
        render_config: RenderConfig = RenderConfig(),
    ):
        # Initialization of environment configurations
        self.config = config
        self.scene_config = scene_config
        self.num_worlds = scene_config.num_scenes
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.num_stack = num_stack

        # Environment parameter setup
        params = self._setup_environment_parameters()
            
        # Initialize simulator with parameters
        self.sim = self._initialize_simulator(params, scene_config)
        
        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs(reset=True).shape[-1],)
        )
        
        self.info_dim = 5  # Number of info features
        self.episode_len = self.config.episode_len
        
        # Rendering setup
        self.visualizer = self._setup_rendering()

    def reset(self):
        """Reset the worlds and return the initial observations."""
        self.sim.reset(list(range(self.num_worlds)))
        self.expert_action, _, _ = self.get_expert_actions()
        return self.get_obs(reset=True)

    def get_dones(self):
        return self.sim.done_tensor().to_torch().squeeze(dim=2).to(torch.float)

    def get_infos(self):
        return (
            self.sim.info_tensor()
            .to_torch()
            .squeeze(dim=2)
            .to(torch.float)
            .to(self.device)
        )
        
    def get_other_infos(self, step):
        _, o, _, d = self.expert_action.shape  # (batch_size, agents, timesteps, dimensions)
        step_expert_action = self.expert_action[:, :, step, :]  # (b, o, 3)
        step_expert_action_expanded = step_expert_action.unsqueeze(1).expand(-1, o, -1, -1)  # (b, o, o, d)
        filtered_partner_id = self.partner_id.clone() 

        gathered_actions = torch.gather(
            step_expert_action_expanded,
            2,  # Gather along the second dimension of agents
            filtered_partner_id.clamp(min=0).long().unsqueeze(-1).expand(-1, -1, -1, d)  # Expand partner_id for features
        )
        gathered_actions[self.partner_mask == 0] = 0
        
        return gathered_actions


    def get_rewards(
        self, collision_weight=0, goal_achieved_weight=1.0, off_road_weight=0
    ):
        """Obtain the rewards for the current step.
        By default, the reward is a weighted combination of the following components:
        - collision
        - goal_achieved
        - off_road

        The importance of each component is determined by the weights.
        """
        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            # Return the weighted combination of the reward components
            info_tensor = self.sim.info_tensor().to_torch()
            off_road = info_tensor[:, :, 0].to(torch.float)

            # True if the vehicle collided with another road object
            # (i.e. a cyclist or pedestrian)
            collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)

            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            return weighted_rewards

    def _copy_actions_to_simulator(self, actions):
        """Copy the provived actions to the simulator."""
        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            # Action space: (acceleration, steering, heading)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "delta_local":
            # Action space: (dx, dy, dyaw)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "state":
            # Following the StateAction struct in types.hpp
            # Need to provide: (x, y, z, yaw, velocity x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
            self.sim.action_tensor().to_torch()[:, :, :10].copy_(actions)
        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )
    
    def get_ids(self):
        return self.ego_id.clone(), self.partner_id.clone()
    
    def get_obs(self, reset=False):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        # EGO STATE
        if self.config.ego_state:
            ego_states_unprocessed = (
                self.sim.self_observation_tensor().to_torch()
            )
            # Omit vehicle ids (last feature)
            self.ego_id = ego_states_unprocessed[:, :, -1]
            ego_states_unprocessed = ego_states_unprocessed[:, :, :-1]

            # Normalize
            if self.config.norm_obs:
                ego_states = self.normalize_ego_state(ego_states_unprocessed)
            else:
                ego_states = ego_states_unprocessed
        else:
            ego_states = torch.Tensor().to(self.device)

        # PARTNER OBSERVATIONS
        if self.config.partner_obs:
            partner_observations = (
                self.sim.partner_observations_tensor().to_torch()
            )
            # Omit vehicle ids (last feature)
            self.partner_id = partner_observations[:, :, :, -1]
            partner_observations = partner_observations[:, :, :, :-1]
            if self.config.norm_obs:  # Normalize observations and then flatten
                partner_observations = self.normalize_and_flatten_partner_obs(
                    partner_observations
                )
            else:  # Flatten along the last two dimensions
                partner_observations = partner_observations.flatten(
                    start_dim=2
                )
        else:
            partner_observations = torch.Tensor().to(self.device)

        # ROAD MAP OBSERVATIONS
        if self.config.road_map_obs:

            road_map_observations_unprocessed = (
                self.sim.agent_roadmap_tensor().to_torch()
            )

            if self.config.norm_obs:
                road_map_observations = self.normalize_and_flatten_map_obs(
                    road_map_observations_unprocessed
                )
            else:
                road_map_observations = (
                    road_map_observations_unprocessed.flatten(start_dim=2)
                )
        else:
            road_map_observations = torch.Tensor().to(self.device)

        # LIDAR OBSERVATIONS
        if self.config.lidar_obs:
            lidar_obs = (
                self.sim.lidar_tensor()
                .to_torch()
                .flatten(start_dim=2, end_dim=-1)
                .to(self.device)
            )
        else:
            # Create empty lidar observations (num_lidar_samples, 4)
            lidar_obs = torch.Tensor().to(self.device)

        # Combine the observations
        obs_filtered = torch.cat(
            (
                ego_states,
                partner_observations,
                road_map_observations,
                lidar_obs,
            ),
            dim=-1,
        )
        curr_cont_mask = self.get_controlled_agents_mask()
        curr_partner_mask = self.get_partner_mask()
        curr_road_mask = self.get_road_mask()
        if self.num_stack > 1:
            if reset:
                stacked_prev_obs = torch.zeros_like(torch.cat([obs_filtered for _ in range(self.num_stack - 1)],dim=-1))
                stacked_prev_cont_mask = torch.zeros_like(torch.cat([self.get_controlled_agents_mask() for _ in range(self.num_stack - 1)],dim=-1))
                stacked_prev_partner_mask = torch.zeros_like(torch.cat([self.get_partner_mask() for _ in range(self.num_stack - 1)],dim=-1))
                stacked_prev_road_mask = torch.zeros_like(torch.cat([self.get_road_mask() for _ in range(self.num_stack - 1)],dim=-1))
            else:
                stacked_prev_obs = self.stacked_obs[..., obs_filtered.shape[-1]:]
                stacked_prev_cont_mask = self.stacked_control_mask[..., self.get_controlled_agents_mask().shape[-1]:]
                stacked_prev_partner_mask = self.stacked_partner_mask[..., self.get_partner_mask().shape[-1]:]
                stacked_prev_road_mask = self.stacked_road_mask[..., self.get_road_mask().shape[-1]:]
            self.stacked_obs = torch.cat([stacked_prev_obs, obs_filtered], dim=-1)
            self.stacked_control_mask = torch.cat([stacked_prev_cont_mask, curr_cont_mask], dim=-1)
            self.stacked_partner_mask = torch.cat([stacked_prev_partner_mask, curr_partner_mask], dim=-1)
            self.stacked_road_mask = torch.cat([stacked_prev_road_mask, curr_road_mask], dim=-1)
        else:
            self.stacked_obs = obs_filtered
            self.stacked_control_mask = curr_cont_mask
            self.stacked_partner_mask = curr_partner_mask
            self.stacked_road_mask = curr_road_mask

        return self.stacked_obs.clone()

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (self.sim.controlled_state_tensor().to_torch() == 1).squeeze(axis=2)

    def get_stacked_controlled_agents_mask(self):
        """Get the stacked control mask."""
        return self.stacked_control_mask.clone()
    
    def get_partner_mask(self):
        """Get the mask for partner observations."""
        return self.partner_mask.clone()

    def get_stacked_partner_mask(self):
        """Get the stacked partner mask."""
        return self.stacked_partner_mask.clone()
    
    def get_road_mask(self):
        """Get the mask for road observations."""
        return self.road_mask.clone()
    
    def get_stacked_road_mask(self):
        """Get the stacked road mask."""
        return self.stacked_road_mask.clone()
    
    def get_partner_goal(self):
        "Get the partner goal"
        return self.partner_goal_state.clone()

    def normalize_ego_state(self, state):
        """Normalize ego state features."""
        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= constants.MAX_SPEED
        state[:, :, 1] /= constants.MAX_VEH_LEN
        state[:, :, 2] /= constants.MAX_VEH_WIDTH

        # Relative goal coordinates for other info
        self.partner_goal_state = state[:, :, 3:5].clone()
        self.partner_goal_state[:, :, 0] = self.normalize_tensor(
            self.partner_goal_state[:, :, 0],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )
        self.partner_goal_state[:, :, 1] = self.normalize_tensor(
            self.partner_goal_state[:, :, 1],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )

        state[:, :, 3] = self.normalize_tensor(
            state[:, :, 3],
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )
        state[:, :, 4] = self.normalize_tensor(
            state[:, :, 4],
            # do the same
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )
        # Uncommment this to exclude the collision state
        # (1 if vehicle is in collision, 1 otherwise)
        # state = state[:, :, :5]

        return state

    def get_expert_actions(self, debug_world_idx=None, debug_veh_idx=None):
        """Get expert actions for the full trajectories across worlds."""

        expert_traj = self.sim.expert_trajectory_tensor().to_torch()
        # Global positions
        positions = expert_traj[:, :, : 2 * self.episode_len].view(
            self.num_worlds, self.max_agent_count, self.episode_len, -1
        )

        # Global velocity
        velocity = expert_traj[
            :, :, 2 * self.episode_len : 4 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        headings = expert_traj[
            :, :, 4 * self.episode_len : 5 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        inferred_expert_actions = expert_traj[
            :, :, 6 * self.episode_len : 16 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        if self.config.dynamics_model == "delta_local":
            inferred_expert_actions = inferred_expert_actions[..., :3]
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
            )
            inferred_expert_actions[..., 1] = torch.clamp(
                inferred_expert_actions[..., 1], -6, 6
            )
            inferred_expert_actions[..., 2] = torch.clamp(
                inferred_expert_actions[..., 2], -3.14, 3.14
            )
        elif self.config.dynamics_model == "state":
            # Extract (x, y, yaw, velocity x, velocity y)
            inferred_expert_actions = torch.cat(
                (
                    positions,  # xy
                    torch.ones((*positions.shape[:-1], 1), device=self.device),
                    headings,  # float (yaw)
                    velocity,  # xy velocity
                    torch.zeros(
                        (*positions.shape[:-1], 4), device=self.device
                    ),
                ),
                dim=-1,
            )
        else:  # classic or bicycle
            inferred_expert_actions = inferred_expert_actions[..., :3]
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
            )
            inferred_expert_actions[..., 1] = torch.clamp(
                inferred_expert_actions[..., 1], -0.3, 0.3
            )
        
        velo2speed = None
        debug_positions = None
        if debug_world_idx is not None and debug_veh_idx is not None:
            velo2speed = (
                torch.norm(velocity[debug_world_idx, debug_veh_idx], dim=-1)
                / constants.MAX_SPEED
            )
            positions[..., 0] = self.normalize_tensor(
                positions[..., 0],
                constants.MIN_REL_AGENT_POS,
                constants.MAX_REL_AGENT_POS,
            )
            positions[..., 1] = self.normalize_tensor(
                positions[..., 1],
                constants.MIN_REL_AGENT_POS,
                constants.MAX_REL_AGENT_POS,
            )
            debug_positions = positions[debug_world_idx, debug_veh_idx]

        return inferred_expert_actions, velo2speed, debug_positions

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: torch.Tensor of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # TODO: Fix (there should not be nans in the obs)
        obs = torch.nan_to_num(obs, nan=0)

        # Speed
        obs[:, :, :, 0] /= constants.MAX_SPEED

        # Relative position
        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )
        obs[:, :, :, 2] = self.normalize_tensor(
            obs[:, :, :, 2],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )

        # Orientation (heading)
        obs[:, :, :, 3] /= constants.MAX_ORIENTATION_RAD

        # Vehicle length and width
        obs[:, :, :, 4] /= constants.MAX_VEH_LEN
        obs[:, :, :, 5] /= constants.MAX_VEH_WIDTH

        # One-hot encode the type of the other visible objects
        one_hot_encoded_object_types = self.one_hot_encode_object_type(
            obs[:, :, :, 6]
        )
        # Concat the one-hot encoding with the rest of the features
        obs = torch.concat((obs[:, :, :, :6], one_hot_encoded_object_types), dim=-1)

        filtered_partner_id = self.partner_id.clone() 
        cont_mask = self.cont_agent_mask.clone()
        filtered_partner_id[..., 1:][filtered_partner_id[..., 1:] <= 0] = -2
        filtered_partner_id[..., 0][filtered_partner_id[..., 0] < 0] = -2
        b, o, _ = filtered_partner_id.shape
        not_existed = filtered_partner_id == -2
        partner_mask_values = torch.gather(
        ~cont_mask.unsqueeze(1).expand(-1, o, -1),  # Expand to (b, o, o)
        2,
        filtered_partner_id.long().clamp(min=0, max=o - 1),  # Clamp invalid indices
        ).long()
        partner_mask2 = ((obs.sum(-1) == 0) | (obs.sum(-1) == 1))
        not_existed = torch.logical_or(not_existed, partner_mask2)
        partner_mask_values[not_existed] = 2
        self.partner_mask = partner_mask_values
        new_mask = torch.where(partner_mask_values == 2, 1, 0)
        # print((new_mask - partner_mask2.int()).abs().sum())
        return obs.flatten(start_dim=2)

    def one_hot_encode_roadpoints(self, roadmap_type_tensor):
        # Set garbage object types to zero
        road_types = torch.where(
            (roadmap_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (roadmap_type_tensor > self.ROAD_MAP_OBJECT_TYPES),
            0.0,
            roadmap_type_tensor,
        ).int()

        return torch.nn.functional.one_hot(
            road_types.long(),
            num_classes=self.ROAD_MAP_OBJECT_TYPES,
        )

    def one_hot_encode_object_type(self, object_type_tensor):
        """One-hot encode the object type."""

        VEHICLE = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Vehicle]
        PEDESTRIAN = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Pedestrian]
        CYCLIST = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Cyclist]
        PADDING = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType._None]

        # Set garbage object elements to zero
        object_types = torch.where(
            (object_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (object_type_tensor > self.MAX_OBJ_ENTITY_ENUM),
            0.0,
            object_type_tensor,
        ).int()

        one_hot_object_type = torch.nn.functional.one_hot(
            torch.where(
                condition=(object_types == VEHICLE)
                | (object_types == PEDESTRIAN)
                | (object_types == CYCLIST)
                | object_types
                == PADDING,
                input=object_types,
                other=0,
            ).long(),
            num_classes=self.ROAD_OBJECT_TYPES,
        )
        return one_hot_object_type

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""
        # Road point coordinates
        obs[:, :, :, 0] = self.normalize_tensor(
            obs[:, :, :, 0],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )
        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )
        
        # Road line segment length
        obs[:, :, :, 2] /= constants.MAX_ROAD_LINE_SEGMENT_LEN

        # Road scale (width and height)
        obs[:, :, :, 3] /= constants.MAX_ROAD_SCALE
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, :, 5] /= constants.MAX_ORIENTATION_RAD
        
        # Road types: one-hot encode them
        one_hot_road_types = self.one_hot_encode_roadpoints(obs[:, :, :, 6])

        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.cat((obs[:, :, :, :6], one_hot_road_types), dim=-1)
        # Get Mask for sorted road points
        #TODO: obs.sum(-1) == 0 is not a good mask
        self.road_mask = ((obs.sum(-1) == 0) | (obs.sum(-1) == 1))
        
        return obs.flatten(start_dim=2)


class GPUDriveDiscreteEnv(GPUDriveTorchEnv):
    """Discrete Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        num_stack=1,
        render_config: RenderConfig = RenderConfig(),
    ):
        super().__init__(
            config=config,
            scene_config=scene_config,
            max_cont_agents=max_cont_agents,
            device=device,
            num_stack=num_stack,
            render_config=render_config,
        )

        self.action_space = self._set_discrete_action_space()

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space based on dynamics model."""
        products = None

        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            products = product(self.dx, self.dy, self.dyaw)
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = self.config.head_tilt_actions.to(self.device)
            products = product(
                self.accel_actions, self.steer_actions, self.head_actions
            )
        elif self.config.dynamics_model == "state":
            self.x = self.config.x.to(self.device)
            self.y = self.config.y.to(self.device)
            self.yaw = self.config.yaw.to(self.device)
            self.vx = self.config.vx.to(self.device)
            self.vy = self.config.vy.to(self.device)

        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        if products is not None:
            for action_idx, (action_1, action_2, action_3) in enumerate(
                products
            ):
                self.action_key_to_values[action_idx] = [
                    action_1.item(),
                    action_2.item(),
                    action_3.item(),
                ]
                self.values_to_action_key[
                    round(action_1.item(), 3),
                    round(action_2.item(), 3),
                    round(action_3.item(), 3),
                ] = action_idx

            self.action_keys_tensor = torch.tensor(
                [
                    self.action_key_to_values[key]
                    for key in sorted(self.action_key_to_values.keys())
                ]
            ).to(self.device)

            return Discrete(n=int(len(self.action_key_to_values)))
        else:
            return Discrete(n=1)

    def step_dynamics(self, actions, use_indices=True):
        if actions is not None:
            if use_indices:
                actions = actions.squeeze(dim=2).long().to(self.device) if actions.dim() == 3 else actions.long().to(self.device)
                action_value_tensor = self.action_keys_tensor[actions]
            else:
                action_value_tensor = torch.nan_to_num(actions, nan=0).float().to(self.device)

            # Feed the action values to gpudrive
            self._copy_actions_to_simulator(action_value_tensor)

        self.sim.step()


class GPUDriveMultiDiscreteEnv(GPUDriveTorchEnv):
    """Multi-Discrete Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        num_stack=1,
        render_config: RenderConfig = RenderConfig(),
    ):
        super().__init__(
            config=config,
            scene_config=scene_config,
            max_cont_agents=max_cont_agents,
            device=device,
            num_stack=num_stack,
            render_config=render_config,
        )

        self.action_space = self._set_multi_discrete_action_space()

    def _set_multi_discrete_action_space(self) -> None:
        """Configure the multi discrete action space."""
        if self.config.dynamics_model == 'delta_local':
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_indices = product(range(len(self.dx)),
                                     range(len(self.dy)),
                                     range(len(self.dyaw)))
            action_values = product(self.dx, self.dy, self.dyaw)
            action_range = [len(self.dx), len(self.dy), len(self.dyaw)]
        else:
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_indices = product(range(len(self.accel_actions)),
                                     range(len(self.steer_actions)),
                                     range(len(self.head_actions)))
            action_values = product(self.accel_actions, self.steer_actions, self.head_actions)
            action_range = [len(self.accel_actions), len(self.steer_actions), len(self.head_actions)]

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        self.action_keys_tensor = torch.zeros(*action_range, 3).to(self.device)

        for action_idx, (action_1, action_2, action_3) in zip(action_indices, action_values):
            action_idx = tuple(action_idx)
            self.action_key_to_values[action_idx] = [
                action_1.item(),
                action_2.item(),
                action_3.item(),
            ]
            self.values_to_action_key[
                round(action_1.item(), 3),
                round(action_2.item(), 3),
                round(action_3.item(), 3),
            ] = action_idx
            self.action_keys_tensor[action_idx] = torch.tensor([action_1, action_2, action_3])

        return MultiDiscrete(nvec=action_range)

    def step_dynamics(self, actions, use_indices=True):
        if actions is not None:
            if use_indices:
                actions = actions.squeeze(dim=3).long().to(self.device) if actions.dim() == 4 else actions.long().to(self.device)
                action_value_tensor = self.action_keys_tensor[actions[...,0], actions[...,1], actions[...,2]]
            else:
                action_value_tensor = torch.nan_to_num(actions, nan=0).float().to(self.device)
                
            # Feed the action values to gpudrive
            self._copy_actions_to_simulator(action_value_tensor)

        self.sim.step()


class GPUDriveContinuousEnv(GPUDriveTorchEnv):
    """Continuous Gym Environment that interfaces with the GPU Drive simulator"""
    
    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        num_stack=1,
        render_config: RenderConfig = RenderConfig(),
    ):
        super().__init__(
            config=config,
            scene_config=scene_config,
            max_cont_agents=max_cont_agents,
            device=device,
            num_stack=num_stack,
            render_config=render_config,
        )

        self.action_space = self._set_continuous_action_space()

    def _set_continuous_action_space(self) -> None:
        """Configure the continuous action space."""
        if self.config.dynamics_model == 'delta_local':
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_1 = self.dx.clone().cpu().numpy()
            action_2 = self.dy.clone().cpu().numpy()
            action_3 = self.dyaw.clone().cpu().numpy()
        elif self.config.dynamics_model == "classic":
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()
        else:
            raise ValueError(
                f"Continuous action space is currently not supported for dynamics_model: {self.config.dynamics_model}."
            )

        action_space = Tuple(
            (Box(action_1.min(), action_1.max(), shape=(1,)),
            Box(action_2.min(), action_2.max(), shape=(1,)),
            Box(action_3.min(), action_3.max(), shape=(1,)))
        )
        return action_space

    def step_dynamics(self, actions, use_indices=None):
        if actions is not None:
            action_value_tensor = actions.to(self.device)
        
            # Feed the action values to gpudrive
            self._copy_actions_to_simulator(action_value_tensor)
        
        self.sim.step()
        

if __name__ == "__main__":
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 2

    env_config = EnvConfig(dynamics_model="delta_local")
    render_config = RenderConfig()
    scene_config = SceneConfig("data/processed/examples", NUM_WORLDS)

    # MAKE ENVIRONMENT
    kwargs = {
        "config": env_config,
        "scene_config": scene_config,
        "max_cont_agents": MAX_CONTROLLED_AGENTS,
        "device": "cuda",
        "num_stack": 5,
        "render_config": render_config
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    
    # RUN
    obs = env.reset()
    frames = []

    for step in range(TOTAL_STEPS):
        print(f"Step: {step}")

        # Take a random actions
        rand_action = torch.Tensor(
            [
                [
                    env.action_space.sample()
                    for _ in range(
                        env_config.max_num_agents_in_scene * NUM_WORLDS
                    )
                ]
            ]
        ).reshape(NUM_WORLDS, env_config.max_num_agents_in_scene, -1)
        expert_action, _, _ = env.get_expert_actions()
        # Step the environment
        env.step_dynamics(rand_action)

        frames.append(env.render())
        
        other_infos = env.get_other_infos(step=step)
        infos = env.get_infos()
        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

    # import imageio
    # imageio.mimsave("world1.gif", np.array(frames))

    env.close()
