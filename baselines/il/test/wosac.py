"""Metrics computation for WOSAC realism evaluation. Calculate WOSAC parts are from
https://github.com/Emerge-Lab/PufferDrive/blob/2.0/pufferlib/ocean/benchmark/evaluator.py
"""
import os, sys
sys.path.append(os.getcwd())

import logging
import torch
import numpy as np
import argparse
from tqdm import tqdm

# GPUDrive
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from baselines.il.test.metrics import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_global_infos(global_agent_obs):
    ego_x = global_agent_obs.pos_x
    ego_y = global_agent_obs.pos_y
    ego_heading = global_agent_obs.rotation_angle.unsqueeze(-1)
    ego_xy = torch.cat([ego_x.unsqueeze(-1), ego_y.unsqueeze(-1)], dim=-1)
    return ego_xy, ego_heading

def collect_rollout(env, bc_policy, num_agent):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    global_agent_obs = env.get_global_state()
    ego_xy, ego_heading = get_global_infos(global_agent_obs)
    ego_length = global_agent_obs.vehicle_length[alive_agent_mask]
    ego_width = global_agent_obs.vehicle_width[alive_agent_mask]

    simulated_xy = torch.zeros((num_agent, 91, 2))
    simulated_heading = torch.zeros((num_agent, 91, 1))
    simulated_xy[:, 0] = ego_xy[alive_agent_mask]
    simulated_heading[:, 0] = ego_heading[alive_agent_mask]

    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")

        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]

        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, *_, = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=False)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions
        env.step_dynamics(all_actions)
        obs = env.get_obs()
        dones = env.get_dones()

        if (dead_agent_mask == True).all():
            break
        elif not dones[alive_agent_mask].all():
            global_agent_obs = env.get_global_state()
            ego_xy, ego_heading = get_global_infos(global_agent_obs)
            simulated_xy[:, time_step + 1] = ego_xy[alive_agent_mask]
            simulated_heading[:, time_step + 1] = ego_heading[alive_agent_mask]

    return simulated_xy, simulated_heading, ego_length, ego_width

def run(args, env, bc_policy, dataset, num_rollout=3):
    obs = env.reset()
    scenario_ids = env.get_scenario_ids()
    scenario_ids_list = np.array([v for k, v in sorted(scenario_ids.items())])
    tracks_to_predict = env.get_tracks_to_predict()
    alive_agent_mask = env.cont_agent_mask.clone()
    # road_edge_polylines = env.get_road_edge_polyline()

    num_agent_per_env = alive_agent_mask.sum(-1).detach().cpu().numpy()
    scenario_ids_list = np.repeat(scenario_ids_list, num_agent_per_env) 
    scenario_ids_list = scenario_ids_list[:, None]
    _, expert_xy, _, expert_heading, expert_valids  = env.get_expert_actions() 
    expert_xy = expert_xy[alive_agent_mask].transpose(1, 2).unsqueeze(1)
    expert_heading = expert_heading[alive_agent_mask].transpose(1, 2)
    expert_valids = expert_valids[alive_agent_mask].transpose(1, 2)
    only_tracks_to_predict = tracks_to_predict[alive_agent_mask]
    num_agent = obs[alive_agent_mask].shape[0]
    simulated_xy = torch.zeros((num_agent, num_rollout, 2, 91))
    simulated_heading = torch.zeros((num_agent, num_rollout, 91))
    for n in range(num_rollout):
        rollout_xy, rollout_heading, ego_length, ego_width = collect_rollout(env, bc_policy,num_agent)
        rollout_xy = rollout_xy.transpose(1, 2)
        rollout_heading = rollout_heading.squeeze(-1)
        simulated_xy[:, n] = rollout_xy
        simulated_heading[:, n] = rollout_heading
    # Extract score
    simulated_xy = simulated_xy.detach().cpu().numpy()
    simulated_heading = simulated_heading.detach().cpu().numpy()
    expert_xy = expert_xy.detach().cpu().numpy()
    expert_heading = expert_heading.detach().cpu().numpy()
    expert_valids = expert_valids.detach().cpu().numpy()
    ego_length = ego_length.detach().cpu().numpy()
    ego_width = ego_width.detach().cpu().numpy()
    only_tracks_to_predict = only_tracks_to_predict.bool().detach().cpu().numpy()
    eval_sim_xy = simulated_xy[only_tracks_to_predict]
    eval_sim_heading = simulated_heading[only_tracks_to_predict]
    eval_expert_xy = expert_xy[only_tracks_to_predict]
    eval_expert_heading = expert_heading[only_tracks_to_predict]
    eval_expert_valids = expert_valids[only_tracks_to_predict]
    eval_ego_length = ego_length[only_tracks_to_predict]
    eval_ego_width = ego_width[only_tracks_to_predict]
    eval_scenario_ids = scenario_ids_list[only_tracks_to_predict]

    sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = compute_kinematic_features(eval_sim_xy[:, :, 0], eval_sim_xy[:, :, 1], eval_sim_heading) # (num_rollout, 2, 91)
    ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = compute_kinematic_features(eval_expert_xy[:, :, 0], eval_expert_xy[:, :, 1], eval_expert_heading)
    speed_validity, acceleration_validity = compute_kinematic_validity(eval_expert_valids) # (1, 1, 91)

    sim_signed_distances, sim_collision_per_step, sim_time_to_collision = compute_interaction_features(
            simulated_xy, simulated_heading, scenario_ids_list, ego_length, ego_width, only_tracks_to_predict, device="cuda"
        )
    ref_signed_distances, ref_collision_per_step, ref_time_to_collision = compute_interaction_features(
            expert_xy, expert_heading, scenario_ids_list, ego_length, ego_width, only_tracks_to_predict, device="cuda", valid=expert_valids,
        )
    # TODO: make road_edge_points and calculate
    # sim_distance_to_road_edge, sim_offroad_per_step = metrics.compute_map_features(
    #         eval_sim_xy,
    #         eval_sim_heading,
    #         eval_scenario_ids,
    #         eval_ego_length,
    #         eval_ego_width,
    #         road_edge_polylines,
    #         device=self.device,
    #     )

    # ref_distance_to_road_edge, ref_offroad_per_step = metrics.compute_map_features(
    #         eval_expert_xy,
    #         eval_expert_heading,
    #         eval_scenario_ids,
    #         eval_ego_length,
    #         eval_ego_width,
    #         road_edge_polylines,
    #         device="cuda",
    #         valid=expert_valids,
    #     )
    ade, min_ade = compute_displacement_error(eval_sim_xy[:, :, 0], eval_sim_xy[:, :, 1], eval_expert_xy[:, :, 0], eval_expert_xy[:, :, 1], eval_expert_valids)
    # print(f'Success World idx : ', torch.where(goal_achieved_ep == 1)[0].tolist())

    sim_collision_indication = np.any(np.where(eval_expert_valids, sim_collision_per_step, False), axis=2)
    ref_collision_indication = np.any(np.where(eval_expert_valids, ref_collision_per_step, False), axis=2)
    
    # TODO: 여기서부터 해야함
    linear_speed_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_linear_speed,
            sim_values=sim_linear_speed,
            meta_data=meta_data["linear_speed"]
        )
    linear_accel_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_linear_accel,
            sim_values=sim_linear_accel,
            meta_data=meta_data["linear_acceleration"]
        )
    angular_speed_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_angular_speed,
            sim_values=sim_angular_speed,
            meta_data=meta_data["angular_speed"]
        )
    angular_accel_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_angular_accel,
            sim_values=sim_angular_accel,
            meta_data=meta_data["angular_acceleration"]
        )
    distance_to_nearest_object_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_signed_distances,
            sim_values=sim_signed_distances,
            meta_data=meta_data["distance_to_nearest_object"]
        )
    time_to_collision_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_time_to_collision,
            sim_values=sim_time_to_collision,
            meta_data=meta_data["time_to_collision"]
        )
    # distance_to_road_edge_log_likelihood = log_likelihood_estimate_timeseries(
    #         log_values=ref_distance_to_road_edge,
    #         sim_values=sim_distance_to_road_edge,
    #         meta_data=meta_data["distance_to_road_edge"]
    #     )
    
    # sim_num_offroad = np.sum(sim_offroad_indication, axis=1)
    # ref_num_offroad = np.sum(ref_offroad_indication, axis=1)
    if args.make_csv:
        if not os.path.exists(f"{args.model_path}/{args.sim_agent}"):
            os.makedirs(f"{args.model_path}/{args.sim_agent}")
        csv_path = f"{args.model_path}/{args.sim_agent}/result_{args.partner_portion_test}.csv"
        file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if file_is_empty:
                column_name = "Model,Dataset"
                
                labels = ["Total"]
                metrics = ["Num","OffRoad","VehCollision","Goal","Collision","GoalProgress",]

                for l, label in enumerate(labels):
                    for metric in metrics:
                        if l == 0 and metric == "Num":
                            continue
                        column_name += f",{label}{metric}"
                f.write(column_name + ",\n")
            data = f"{args.model_name},{dataset},{off_road_rate},{veh_coll_rate},{goal_rate},{collision_rate},{goal_progress_ratio},"
            f.write(data + ",\n")

    # return off_road_rate, veh_coll_rate, goal_rate, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    
    parser.add_argument('--dataset-size', type=int, default=10) # total_world
    parser.add_argument('--batch-size', type=int, default=10) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/log_replay_test')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s3_0826_035640.pth') # early_attn_s11_0808_043910
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=1.0)
    parser.add_argument('--sim-agent', '-sa', type=str, default='log_replay', choices=['log_replay', 'self_play', 'delta_replay'])
    parser.add_argument('--dataset', '-d', type=str, default='training', choices=['training', 'validation'])
    args = parser.parse_args()
    # Configurations
    num_cont_agents = 128

    # Create data loader
    if args.dataset == 'training':
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/training/",
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = args.dataset_size
    else:
        # Test Scene
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/validation/",
            batch_size=args.batch_size,
            dataset_size=9987,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = 9987
    print(f'{args.dataset} len scene loader {len(scene_loader)}')
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore',
        num_stack=5

    )
    render_config = RenderConfig(
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=scene_loader,
        max_cont_agents=num_cont_agents,  # Number of agents to control
        device="cuda",
        render_config=render_config,
        action_type="continuous",
    )
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    num_iter = int(dataset_size // args.batch_size) if dataset_size != 0 else 0
    if args.sim_agent == 'log_replay':
        remove_controlled_agents = False
    else:
        remove_controlled_agents = True

    for i in tqdm(range(num_iter)):
        run(args, env, bc_policy, dataset=args.dataset)
        if i != num_iter - 1:
            print('SWAP!!')
            env.swap_data_batch()
    env.close()

