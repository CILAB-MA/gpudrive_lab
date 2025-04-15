"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import os, sys
import numpy as np
from collections import defaultdict
sys.path.append(os.getcwd())
import argparse

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.config import DynamicsModel, ActionSpace
from algorithms.il.model.bc import *
from algorithms.il.analyze.linear_probing.linear_prob import register_all_layers_forward_hook
from pygpudrive.registration import make

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    # ENV
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--start-idx', '-st', type=int, default=0)
    parser.add_argument('--num-world', '-w', type=int, default=2)
    parser.add_argument('--seed', type=int, default=44)
    # EXPERIMENT
    parser.add_argument('--dataset', type=str, default='valid', choices=['train', 'valid'],)
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/early_attn_all_baseline_0407')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_all_data_0403_132702')
    parser.add_argument('--make-image', '-mv', action='store_true')
    parser.add_argument('--image-path', '-vp', type=str, default='/data/intervention/test')

    # INTERVENTION
    parser.add_argument('--intervention-idx', '-o', type=list, default=[10] * 2) # intervention partner idx
    parser.add_argument('--intervention-label', '-l', type=int, default=10) # change position label
    args = parser.parse_args()
    return args

def fill_ego(partner_idx, ego_attn_score, partner_mask):
    multi_head_num = ego_attn_score.shape[1]
    ego_attn_score = ego_attn_score.transpose(1, 2)
    n, _ = partner_idx.shape
    filled_tensor = torch.full((n, 128, multi_head_num), -1.0).to(args.device)

    row_indices = torch.arange(n).unsqueeze(1).expand_as(partner_idx).to(args.device)
    valid_rows = row_indices[~partner_mask]
    valid_cols = partner_idx[~partner_mask].int()
    valid_values = ego_attn_score[~partner_mask]

    filled_tensor[valid_rows, valid_cols] = valid_values.float()
    
    return filled_tensor

def run(args):
    
    # Configurations
    NUM_WORLDS = args.num_world
    NUM_PARTNER = 128
    MAX_NUM_OBJECTS = 1
    ROLLOUT_LEN = 5

    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
                               num_scenes=NUM_WORLDS,
                               start_idx=args.start_idx,
                               discipline=SelectionDiscipline.RANGE_N)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore'

    )
    render_config = RenderConfig(
        draw_obj_idx=True,
        draw_expert_footprint=True,
        draw_only_ego_footprint=True,
        draw_ego_importance=False,
        draw_other_lp=True,
        draw_other_idx=args.intervention_idx,
        draw_lp_label=False
    )
    # Initialize environment
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "render_config": render_config,
        "max_cont_agents": MAX_NUM_OBJECTS,
        "device": args.device,
        "num_stack": args.num_stack
    }
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    print(f'model: {args.model_path}/{args.model_name}', )
    
    # Load model
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}.pth", weights_only=False).to(args.device)
    bc_policy.eval()
    ro_attn_layers = register_all_layers_forward_hook(bc_policy.ro_attn)
    
    # Load other_lp heads
    other_lp_heads = {}
    other_lp_path = os.path.join(args.model_path, "linear_prob", args.model_name, f"seed{args.seed}")
    for lp in os.listdir(other_lp_path):
        file_name, file_type = lp.split('.')
        if file_type == 'pth' and "early" in file_name:
            other_lp_heads[file_name] = torch.load(os.path.join(other_lp_path, lp), weights_only=False).to(args.device)
            other_lp_heads[file_name].eval()

    # Load ego_lp heads
    ego_lp_heads = {}
    ego_lp_path = os.path.join(args.model_path, "ego_linear_prob", args.model_name, f"seed{args.seed}")
    for lp in os.listdir(ego_lp_path):
        file_name, file_type = lp.split('.')
        if file_type == 'pth' and "final" in file_name:
            ego_lp_heads[file_name] = torch.load(os.path.join(ego_lp_path, lp), weights_only=False).to(args.device)
            ego_lp_heads[file_name].eval()

    # Label viz for image
    if render_config.draw_expert_footprint or render_config.draw_other_aux:
        obs = env.reset()
        expert_actions, _, _ = env.get_expert_actions()
        for time_step in range(env.episode_len):
            for world_render_idx in range(NUM_WORLDS):
                env.save_footprint(world_render_idx=world_render_idx, time_step=time_step)
                env.save_aux(world_render_idx=world_render_idx, time_step=time_step)
            env.step_dynamics(expert_actions[:, :, time_step, :])
            obs = env.get_obs()
            dones = env.get_dones()
            if (dones == True).all():
                break

    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(NUM_WORLDS)]
    expert_actions, _, _ = env.get_expert_actions()
    infos = env.get_infos()

    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to(args.device)
        
        # MASK
        ego_masks = env.get_stacked_controlled_agents_mask().to(args.device)
        partner_masks = env.get_stacked_partner_mask().to(args.device)
        road_masks = env.get_stacked_road_mask().to(args.device)
        ego_masks = ego_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN)
        partner_masks = partner_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        partner_mask_bool = partner_masks == 2


        road_masks = road_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        all_masks = [ego_masks[~dead_agent_mask], partner_mask_bool[~dead_agent_mask], road_masks[~dead_agent_mask]]

        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            is_world_alive = (~dead_agent_mask).sum(axis=-1).bool()
            # Get action
            context, ego_attn_score, _ = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
            
            # Get initial other & ego heads
            other_lp_dict = defaultdict(dict)
            ego_lp_dict = defaultdict(dict)
            ego_lp_prime_dict = defaultdict(dict)
            for (ego_name, ego_head) , (other_name, other_head) in zip(ego_lp_heads.items(), other_lp_heads.items()):
                other_pred = other_head.predict(ro_attn_layers['0'][:,1:,:]) # todo: '0' -> lp layer
                other_pred_weight = other_head.head.weight[args.intervention_label] # todo: thsi should be list! e.g.) 3. 10. 5. 7
                ego_pred = ego_head.predict(ro_attn_layers['0'][:,0,:]) # todo: '0' -> lp layer
                ego_pred_prime = ego_head.predict(ro_attn_layers['0'][:,0,:] + other_pred_weight) # todo: '0' -> lp layer
                ego_world = torch.zeros((NUM_WORLDS, 1)).long().to(args.device)
                ego_world_prime = torch.zeros((NUM_WORLDS, 1)).long().to(args.device)
                ego_world[is_world_alive] = ego_pred.unsqueeze(-1)
                ego_world_prime[is_world_alive] = ego_pred_prime.unsqueeze(-1)
                other_lp_dict[other_head.future_step] = other_pred
                ego_lp_dict[ego_head.future_step] = ego_world
                ego_lp_prime_dict[ego_head.future_step] = ego_world_prime

        all_actions[~dead_agent_mask, :] = actions
        intervention_idx = torch.tensor(args.intervention_idx).long().to('cuda')
        response_type = env.get_response_type_tensor().squeeze(2)
        # intervention index (fig) -> index of 127

        
        moving_mask = (response_type == 0).bool()
        true_indices = moving_mask.nonzero(as_tuple=False)
        cumsum_mask = moving_mask.cumsum(dim=1) - 1
        cumsum_mask[~moving_mask] = -1
        idx_match = (cumsum_mask == intervention_idx.unsqueeze(1))
        matched_col = idx_match.float().argmax(dim=1)
        if args.make_image:
            partner_idx = env.partner_id[~dead_agent_mask].clone()
            if render_config.draw_other_lp:
                for future_step, other_lp in other_lp_dict.items():
                    other_lp = fill_ego(partner_idx, other_lp.unsqueeze(1), partner_mask_bool[:,:,-1,:][~dead_agent_mask])
                    world_other_lp = torch.zeros(NUM_WORLDS, NUM_PARTNER).to(args.device)
                    world_other_lp[(~dead_agent_mask).sum(dim=-1) == 1] = other_lp.squeeze(-1)
                    other_lp_dict[future_step] = world_other_lp[torch.arange(len(intervention_idx)), matched_col].unsqueeze(-1)
                env.save_lp_pred(ego_lp_dict, other_lp_dict, ego_lp_prime_dict, matched_col)

            for world_render_idx in range(NUM_WORLDS):
                if not is_world_alive[world_render_idx]:
                    continue
                frame = env.render(world_render_idx=world_render_idx, time_step=time_step)
                frames[world_render_idx].append(frame)

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()

        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    
    if args.make_image:
        image_path = os.path.join(args.image_path, args.dataset, args.model_name,'intervention')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for world_render_idx in range(NUM_WORLDS):
            for time_step, frame in enumerate(frames[world_render_idx]):
                filename = f'scene_{world_render_idx + args.start_idx}_step_{time_step}.jpg'
                filepath = os.path.join(image_path, filename)
                imageio.imwrite(filepath, frame)
    print('finish!')
if __name__ == "__main__":
    args = parse_args()
    run(args)