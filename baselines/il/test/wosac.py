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
import pandas as pd
import gc
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def collect_wosac_random_baseline(env, num_agent, init_step=10):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    global_agent_obs = env.get_global_state()
    ego_xy, ego_heading = get_global_infos(global_agent_obs)
    ego_length = global_agent_obs.vehicle_length[alive_agent_mask]
    ego_width = global_agent_obs.vehicle_width[alive_agent_mask]

    simulated_xy = torch.zeros((num_agent, 81, 2)).to("cuda")
    simulated_heading = torch.zeros((num_agent, 81, 1)).to("cuda")
    infos = env.get_infos()
    goal_achieved_init = infos.goal_achieved[alive_agent_mask]
    simulated_xy[:, 0] = ego_xy[alive_agent_mask]
    simulated_heading[:, 0] = ego_heading[alive_agent_mask]

    # Update using Gaussian:
    samples = torch.normal(mean=1.0, std=0.1, size=(num_agent, 81, 3), device="cuda")
    for time_step in tqdm(range(1, env.episode_len - init_step)):
        dx, dy, d_heading = samples[:, time_step, 0], samples[:, time_step, 1], samples[:, time_step, 2]
        x, y, heading = simulated_xy[:, time_step - 1, 0] , simulated_xy[:, time_step - 1, 1],  simulated_heading[:, time_step - 1]

        cos_h = torch.cos(heading).reshape(-1)
        sin_h = torch.sin(heading).reshape(-1)

        x_n = x + dx * cos_h - dy * sin_h
        y_n = y + dx * sin_h + dy * cos_h
        heading += d_heading.unsqueeze(-1)

        simulated_xy[:, time_step, 0] = x_n
        simulated_xy[:, time_step, 1] = y_n
        simulated_heading[:, time_step] = heading
    return simulated_xy, simulated_heading, ego_length, ego_width, goal_achieved_init.bool()



def get_global_infos(global_agent_obs):
    ego_x = global_agent_obs.pos_x
    ego_y = global_agent_obs.pos_y
    ego_heading = global_agent_obs.rotation_angle.unsqueeze(-1)
    ego_xy = torch.cat([ego_x.unsqueeze(-1), ego_y.unsqueeze(-1)], dim=-1)
    return ego_xy, ego_heading

def collect_rollout(env, bc_policy, num_agent, init_step=10):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    global_agent_obs = env.get_global_state()
    ego_xy, ego_heading = get_global_infos(global_agent_obs)
    ego_length = global_agent_obs.vehicle_length[alive_agent_mask]
    ego_width = global_agent_obs.vehicle_width[alive_agent_mask]
    simulated_xy = torch.zeros((num_agent, 81, 2)).to("cuda")
    simulated_heading = torch.zeros((num_agent, 81, 1)).to("cuda")
    simulated_xy[:, 0] = ego_xy[alive_agent_mask]
    simulated_heading[:, 0] = ego_heading[alive_agent_mask]
    infos = env.get_infos()
    goal_achieved_init = infos.goal_achieved[alive_agent_mask]
    goal_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.float32).to("cuda")
    off_road_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]
    for time_step in tqdm(range(env.episode_len - init_step)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")

        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        
        # Record when agent status changed(goal or collided)
        off_road = infos.off_road[alive_agent_mask]
        veh_collision = infos.collided[alive_agent_mask]
        goal_achieved = infos.goal_achieved[alive_agent_mask]
        goal_mask = (goal_achieved > 0) & (goal_timesteps == -1)
        off_road_mask = (off_road > 0) & (off_road_timesteps == -1)
        off_road_timesteps[off_road_mask] = time_step
        goal_timesteps[goal_mask] = time_step
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
        infos = env.get_infos()
        off_road_ep += infos.off_road[alive_agent_mask]
        veh_collision_ep += infos.collided[alive_agent_mask]
        goal_achieved_ep += infos.goal_achieved[alive_agent_mask]
        off_road_ep = torch.clamp(off_road_ep, max=1)
        veh_collision_ep = torch.clamp(veh_collision_ep, max=1)
        goal_achieved_ep = torch.clamp(goal_achieved_ep, max=1)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            ego_xy_filtered[goal_achieved_ep.bool()] = simulated_xy[goal_achieved_ep.bool(), time_step]
            ego_heading_filtered[goal_achieved_ep.bool()] = simulated_heading[goal_achieved_ep.bool(), time_step]
            rem = 81 - (time_step + 1)
            xy_fill = ego_xy_filtered.unsqueeze(1).expand(-1, rem, -1)  # (N, rem, 2)
            headomg_fill = ego_heading_filtered.unsqueeze(1).expand(-1, rem, -1)  # (N, rem, 2)
            simulated_xy[:, time_step + 1:] = xy_fill
            simulated_heading[:, time_step + 1:] = headomg_fill
            break
        elif not dones[alive_agent_mask].all():
            global_agent_obs = env.get_global_state()
            ego_xy, ego_heading = get_global_infos(global_agent_obs)
            ego_xy_filtered = ego_xy[alive_agent_mask]
            ego_heading_filtered  = ego_heading[alive_agent_mask]
            if goal_achieved_ep.any(): 
                # since gpudrive remove the vehicle after goal arrived, so fix the last pos if reached goal
                ego_xy_filtered[goal_achieved_ep.bool()] = simulated_xy[goal_achieved_ep.bool(), time_step]
                ego_heading_filtered[goal_achieved_ep.bool()] = simulated_heading[goal_achieved_ep.bool(), time_step]
            simulated_xy[:, time_step + 1] = ego_xy_filtered
            simulated_heading[:, time_step + 1] = ego_heading_filtered
    off_road_rate = off_road_ep.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision_ep.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved_ep.sum().float() / alive_agent_mask.sum().float()
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
    print()
    return simulated_xy, simulated_heading, ego_length, ego_width, goal_achieved_init.bool()

def run(args, env, bc_policy, dataset, num_rollout=32):
    obs = env.reset()
    scenario_ids = env.get_scenario_ids()
    scenario_ids_list = np.array([v for k, v in sorted(scenario_ids.items())])
    tracks_to_predict = env.get_tracks_to_predict()
    is_sdc = env.get_is_sdc()
    alive_agent_mask = env.cont_agent_mask.clone()
    
    road_edge_polylines = env.get_road_edge_polyline()
    
    num_agent_per_env = alive_agent_mask.sum(-1).detach().cpu().numpy()
    scenario_ids_agent = np.repeat(scenario_ids_list, num_agent_per_env) 
    scenario_ids_agent = scenario_ids_agent[:, None]

    _, expert_xy, _, expert_heading, expert_valids  = env.get_expert_actions() 
    expert_xy = expert_xy[alive_agent_mask].transpose(1, 2).unsqueeze(1)[..., 10:]
    expert_heading = expert_heading[alive_agent_mask].transpose(1, 2)[..., 10:]
    expert_valids = expert_valids[alive_agent_mask].transpose(1, 2)[..., 10:]
    only_tracks_to_predict = tracks_to_predict[alive_agent_mask]
    is_vehicle = is_sdc[alive_agent_mask].unsqueeze(-1)
    num_agent = obs[alive_agent_mask].shape[0]
    simulated_xy = torch.zeros((num_agent, num_rollout, 2, 81))
    simulated_heading = torch.zeros((num_agent, num_rollout, 81))
    for n in range(num_rollout):
        if args.is_random:
            rollout_xy, rollout_heading, ego_length, ego_width, goal_achieved_init = collect_wosac_random_baseline(env, num_agent)
        else:
            rollout_xy, rollout_heading, ego_length, ego_width, goal_achieved_init = collect_rollout(env, bc_policy, num_agent)
        rollout_xy = rollout_xy.transpose(1, 2)
        rollout_heading = rollout_heading.squeeze(-1)
        simulated_xy[:, n] = rollout_xy
        simulated_heading[:, n] = rollout_heading

        del rollout_xy, rollout_heading
        torch.cuda.empty_cache()
        gc.collect()
    # Extract score
    goal_achieved_init = goal_achieved_init.detach().cpu().numpy()
    simulated_xy = simulated_xy[~goal_achieved_init].detach().cpu().numpy()
    simulated_heading = simulated_heading[~goal_achieved_init].detach().cpu().numpy()
    expert_xy = expert_xy[~goal_achieved_init].detach().cpu().numpy()
    expert_heading = expert_heading[~goal_achieved_init].detach().cpu().numpy()
    expert_valids = expert_valids[~goal_achieved_init].detach().cpu().numpy()
    ego_length = ego_length[~goal_achieved_init].detach().cpu().numpy()
    ego_width = ego_width[~goal_achieved_init].detach().cpu().numpy()
    is_vehicle = is_vehicle[~goal_achieved_init].detach().cpu().numpy()
    only_tracks_to_predict = only_tracks_to_predict[~goal_achieved_init].bool().detach().cpu().numpy()
    scenario_ids_agent = scenario_ids_agent[~goal_achieved_init]
    
    eval_sim_xy = simulated_xy[only_tracks_to_predict]
    eval_sim_heading = simulated_heading[only_tracks_to_predict]
    eval_expert_xy = expert_xy[only_tracks_to_predict]
    eval_expert_heading = expert_heading[only_tracks_to_predict]
    eval_expert_valids = expert_valids[only_tracks_to_predict]
    eval_ego_length = ego_length[only_tracks_to_predict]
    eval_ego_width = ego_width[only_tracks_to_predict]
    eval_scenario_ids = scenario_ids_agent[only_tracks_to_predict]
    eval_is_vehicle = is_vehicle[only_tracks_to_predict]

    sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = compute_kinematic_features(
        eval_sim_xy[:, :, 0], eval_sim_xy[:, :, 1], eval_sim_heading) # (num_rollout, 2, 81)
    ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = compute_kinematic_features(
        eval_expert_xy[:, :, 0], eval_expert_xy[:, :, 1], eval_expert_heading)
    
    speed_validity, acceleration_validity = compute_kinematic_validity(expert_valids[only_tracks_to_predict]) # (1, 1, 81)

    sim_signed_distances, sim_collision_per_step, sim_time_to_collision = compute_interaction_features(
            simulated_xy, simulated_heading, scenario_ids_agent, ego_length, ego_width, only_tracks_to_predict, device="cuda"
        )
    
    ref_signed_distances, ref_collision_per_step, ref_time_to_collision = compute_interaction_features(
            expert_xy, expert_heading, scenario_ids_agent, ego_length, ego_width, only_tracks_to_predict, device="cuda", valid=expert_valids,
        )
    
    sim_distance_to_road_edge, sim_offroad_per_step = compute_map_features(
            eval_sim_xy,
            eval_sim_heading,
            eval_scenario_ids,
            eval_ego_length,
            eval_ego_width,
            road_edge_polylines,
            device="cuda",
        )

    ref_distance_to_road_edge, ref_offroad_per_step = compute_map_features(
            eval_expert_xy,
            eval_expert_heading,
            eval_scenario_ids,
            eval_ego_length,
            eval_ego_width,
            road_edge_polylines,
            device="cuda",
            valid=eval_expert_valids,
        )
    
    ade, min_ade = compute_displacement_error(eval_sim_xy[:, :, 0], eval_sim_xy[:, :, 1], eval_expert_xy[:, :, 0], eval_expert_xy[:, :, 1], eval_expert_valids)

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
    distance_to_road_edge_log_likelihood = log_likelihood_estimate_timeseries(
            log_values=ref_distance_to_road_edge,
            sim_values=sim_distance_to_road_edge,
            meta_data=meta_data["distance_to_road_edge"]
        )
    
    speed_log_likelihood = reduce_average_with_validity(
        linear_speed_log_likelihood,
        speed_validity[:, 0, :],
        axis=1,
    )

    accel_log_likelihood = reduce_average_with_validity(
        linear_accel_log_likelihood,
        acceleration_validity[:, 0, :],
        axis=1,
    )

    angular_speed_log_likelihood = reduce_average_with_validity(
        angular_speed_log_likelihood,
        speed_validity[:, 0, :],
        axis=1,
    )

    angular_accel_log_likelihood = reduce_average_with_validity(
        angular_accel_log_likelihood,
        acceleration_validity[:, 0, :],
        axis=1,
    )

    distance_to_nearest_object_log_likelihood = reduce_average_with_validity(
        distance_to_nearest_object_log_likelihood,
        eval_expert_valids[:, 0, :],
        axis=1,
    )

    # TTC is computed only for vehicles
    ttc_valid = eval_expert_valids & eval_is_vehicle[..., None].astype("bool")
    time_to_collision_log_likelihood = reduce_average_with_validity(
        time_to_collision_log_likelihood,
        ttc_valid[:, 0, :],
        axis=1,
    )

    distance_to_road_edge_log_likelihood = reduce_average_with_validity(
        distance_to_road_edge_log_likelihood,
        eval_expert_valids[:, 0, :],
        axis=1,
    )

    sim_collision_indication = np.any(np.where(eval_expert_valids, sim_collision_per_step, False), axis=2)
    ref_collision_indication = np.any(np.where(eval_expert_valids, ref_collision_per_step, False), axis=2)

    sim_num_collisions = np.sum(sim_collision_indication, axis=1)
    ref_num_collisions = np.sum(ref_collision_indication, axis=1)

    collision_log_likelihood = log_likelihood_estimate_scenario_level(
            log_values=ref_collision_indication[:, 0],
            sim_values=sim_collision_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )
    
    sim_offroad_indication = np.any(np.where(eval_expert_valids, sim_offroad_per_step, False), axis=2)
    ref_offroad_indication = np.any(np.where(eval_expert_valids, ref_offroad_per_step, False), axis=2)

    sim_num_offroad = np.sum(sim_offroad_indication, axis=1)
    ref_num_offroad = np.sum(ref_offroad_indication, axis=1)
    offroad_log_likelihood = log_likelihood_estimate_scenario_level(
                log_values=ref_offroad_indication[:, 0],
                sim_values=sim_offroad_indication,
                min_val=0.0,
                max_val=1.0,
                num_bins=2,
                use_bernoulli=True,
            )
    
    # eval_agent_ids = ground_truth_trajectories["id"][only_tracks_to_predict]

    df = pd.DataFrame(
        {
            # "agent_id": eval_agent_ids.flatten(),
            "scenario_id": eval_scenario_ids.flatten(),
            "num_collisions_sim": sim_num_collisions.flatten(),
            "num_collisions_ref": ref_num_collisions.flatten(),
            "num_offroad_sim": sim_num_offroad.flatten(),
            "num_offroad_ref": ref_num_offroad.flatten(),
            "ade": ade,
            "min_ade": min_ade,
            "likelihood_linear_speed": speed_log_likelihood,
            "likelihood_linear_acceleration": accel_log_likelihood,
            "likelihood_angular_speed": angular_speed_log_likelihood,
            "likelihood_angular_acceleration": angular_accel_log_likelihood,
            "likelihood_distance_to_nearest_object": distance_to_nearest_object_log_likelihood,
            "likelihood_time_to_collision": time_to_collision_log_likelihood,
            "likelihood_collision_indication": collision_log_likelihood,
            "likelihood_distance_to_road_edge": distance_to_road_edge_log_likelihood,
            "likelihood_offroad_indication": offroad_log_likelihood,
        }
    )

    scene_level_results = df.groupby("scenario_id")[
        [
            "ade",
            "min_ade",
            "num_collisions_sim",
            "num_collisions_ref",
            "num_offroad_sim",
            "num_offroad_ref",
            "likelihood_linear_speed",
            "likelihood_linear_acceleration",
            "likelihood_angular_speed",
            "likelihood_angular_acceleration",
            "likelihood_distance_to_nearest_object",
            "likelihood_time_to_collision",
            "likelihood_collision_indication",
            "likelihood_distance_to_road_edge",
            "likelihood_offroad_indication",
        ]
    ].mean()
    
    likelihood_cols = [c for c in scene_level_results.columns if "likelihood" in c]
    scene_level_results[likelihood_cols] = np.exp(scene_level_results[likelihood_cols])
    
    scene_level_results["realism_meta_score"] = scene_level_results.apply(compute_metametric, axis=1)
    scene_level_results["num_agents"] = df.groupby("scenario_id").size()
    scene_level_results = scene_level_results[
        ["num_agents"] + [col for col in scene_level_results.columns if col != "num_agents"]
    ]
    kin_cols = [
        "likelihood_linear_speed",
        "likelihood_linear_acceleration",
        "likelihood_angular_speed",
        "likelihood_angular_acceleration",
    ]
    int_cols = [
        "likelihood_distance_to_nearest_object",
        "likelihood_time_to_collision",
        "likelihood_collision_indication",
    ]
    map_cols = [
        "likelihood_distance_to_road_edge",
        "likelihood_offroad_indication",
    ]

    scene_level_results["kinematic_metrics"] = scene_level_results[kin_cols].mean(axis=1)
    scene_level_results["interactive_metrics"] = scene_level_results[int_cols].mean(axis=1)
    scene_level_results["map_based_metrics"] = scene_level_results[map_cols].mean(axis=1)
    aggregate_results = True
    if aggregate_results:
        aggregate_metrics = scene_level_results.mean().to_dict()
        aggregate_metrics["total_num_agents"] = scene_level_results["num_agents"].sum()
        # Convert numpy types to Python native types
        return {k: v.item() if hasattr(v, "item") else v for k, v in aggregate_metrics.items()}
    else:
        print("\n Scene-level results:\n")
        print(scene_level_results)

        print(f"\n Overall realism meta score: {scene_level_results['realism_meta_score'].mean():.4f}")
        print(f"\n Overall minADE: {scene_level_results['min_ade'].mean():.4f}")
        print(f"\n Overall ADE: {scene_level_results['ade'].mean():.4f}")

        # print(f"\n Full agent-level results:\n")
        # print(df)
        return scene_level_results


    # return off_road_rate, veh_coll_rate, goal_rate, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    
    parser.add_argument('--dataset-size', type=int, default=50) # total_world
    parser.add_argument('--batch-size', type=int, default=50) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_100')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s3_0901_064245.pth')
    parser.add_argument('--is-random', '-r', action='store_true')
    parser.add_argument('--sim-agent', '-sa', type=str, default='log_replay', choices=['log_replay', 'self_play', 'delta_replay'])
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    parser.add_argument('--init-steps', type=int, default=11)
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
            dataset_size=1000,
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
        num_stack=5,
        remove_non_vehicles=False,
        init_steps=args.init_steps
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
    import json
    from pathlib import Path
    p = Path(args.model_path)
    parts = p.resolve().parts
    name = p.name + f"_{args.model_name[11:13]}" if not args.is_random else "random"
    for i in tqdm(range(num_iter)):
        results = run(args, env, bc_policy, dataset=args.dataset)
        with open(f"/data/full_version/exp_{name}.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(results, ensure_ascii=False))
            f.write("\n")
        if i != num_iter - 1:
            print('SWAP!!')
            env.swap_data_batch()
    env.close()

