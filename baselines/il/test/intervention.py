"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging, imageio
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapy as media
import argparse, functools
from collections import defaultdict
# GPUDrive
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from collections import OrderedDict
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/data_cut_add')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_seed_3_0523_204605.pth')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-image', '-mi', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument("--action-image-dir", "-aid", type=str, default="/data/full_version/intervention/training")
    parser.add_argument('--dataset-size', type=int, default=5) # total_world
    parser.add_argument('--batch-size', type=int, default=5) # num_world
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=1)
    # INTERVENTION
    parser.add_argument('--intervention-idx', '-i', type=list, default=[2, 2, 10, 2, 1]) # intervention partner idx
    parser.add_argument('--scene-idx', '-si', type=list, default=[282, 287, 330, 335, 349]) # intervention partner idx
    parser.add_argument('--intervention-label', '-l', type=list, default=[10] * 5) # change position label
    args = parser.parse_args()
    return args

def fill_ego(partner_idx, importance_score, partner_mask):
    n, _ = partner_idx.shape
    filled_tensor = torch.full((n, 127), -1.0).to("cuda")

    row_indices = torch.arange(n).unsqueeze(1).expand_as(partner_idx).to("cuda")
    valid_rows = row_indices[~partner_mask]
    valid_cols = partner_idx[~partner_mask].int()
    valid_values = importance_score[~partner_mask]

    filled_tensor[valid_rows, valid_cols] = valid_values.float()
    
    return filled_tensor

def register_all_layers_forward_hook(model):
    hidden_vector_dict = OrderedDict()

    def hook_fn(module, input, output, name):
        try:
            hidden_vector_dict[name] = output.detach()
        except AttributeError:
            hidden_vector_dict[name] = output['last_hidden_state'].detach()

    def _register(module, prefix=""):
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer.register_forward_hook(functools.partial(hook_fn, name=full_name))

            _register(layer, full_name)

    _register(model)

    return hidden_vector_dict

def run(args):

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    action_image_dir = args.action_image_dir
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore',
        num_stack=5

    )
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        shuffle=False
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=args.max_cont_agents,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    torch.set_printoptions(precision=3)
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    num_iter = int(args.dataset_size // args.batch_size)
    ro_attn_layers = register_all_layers_forward_hook(bc_policy.ro_attn)
    fu_attn_layers = register_all_layers_forward_hook(bc_policy.fusion_attn)
    
    # Load other_lp heads
    other_lp_heads = {}
    other_lp_path = os.path.join(args.model_path, "other_linear_prob", args.model_name[:-4], f"seed{args.seed}")
    for lp in os.listdir(other_lp_path):
        if 'pth' not in lp:
            continue
        file_name, file_type = lp.split('.')
        if file_type == 'pth' and "early" in file_name:
            other_lp_heads[file_name] = torch.load(os.path.join(other_lp_path, lp), weights_only=False).to("cuda")
            other_lp_heads[file_name].eval()

    # Load ego_lp heads
    ego_lp_heads = {}
    ego_lp_path = os.path.join(args.model_path, "ego_linear_prob", args.model_name[:-4], f"seed{args.seed}")
    for lp in os.listdir(ego_lp_path):
        if 'pth' not in lp:
            continue
        file_name, file_type = lp.split('.')
        if file_type == 'pth' and "final" in file_name:
            ego_lp_heads[file_name] = torch.load(os.path.join(ego_lp_path, lp), weights_only=False).to("cuda")
            ego_lp_heads[file_name].eval()

    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.dataset_size)]
    expert_actions, _, _, _, _ = env.get_expert_actions()
    infos = env.get_infos()
    intervention_label = torch.from_numpy(np.array(args.intervention_label)).to("cuda")
    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2

        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]

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
                other_pred_weight = other_head.head.weight[intervention_label[is_world_alive]] # todo: thsi should be list! e.g.) 3. 10. 5. 7
                ego_pred = ego_head.predict(ro_attn_layers['0'][:,0,:]) # todo: '0' -> lp layer
                ego_pred_prime = ego_head.predict(ro_attn_layers['0'][:,0,:] + other_pred_weight) # todo: '0' -> lp layer
                ego_world = torch.zeros((args.dataset_size, 1)).long().to("cuda")
                ego_world_prime = torch.zeros((args.dataset_size, 1)).long().to("cuda")
                ego_world[is_world_alive] = ego_pred.unsqueeze(-1)
                ego_world_prime[is_world_alive] = ego_pred_prime.unsqueeze(-1)
                other_lp_dict[other_head.future_step] = other_pred
                ego_lp_dict[ego_head.future_step] = ego_world
                ego_lp_prime_dict[ego_head.future_step] = ego_world_prime

        all_actions[~dead_agent_mask, :] = actions
        intervention_idx = torch.tensor(args.intervention_idx).long().to('cuda')
        response_type = env.sim.response_type_tensor().to_torch().squeeze(2)
        # intervention index (fig) -> index of 127
        
        moving_mask = (response_type == 0).bool()
        true_indices = moving_mask.nonzero(as_tuple=False)
        cumsum_mask = moving_mask.cumsum(dim=1) - 1
        cumsum_mask[~moving_mask] = -1
        idx_match = (cumsum_mask == intervention_idx.unsqueeze(1))
        matched_col = idx_match.float().argmax(dim=1)
        if args.make_image:
            partner_idx = env.partner_ids[~dead_agent_mask].clone()
            for future_step, other_lp in other_lp_dict.items():
                # ------------------여기부터 작업해야 함 아이디 ㅅㅂ것 매칭이 안돼고 지라라리 ㅇ나 ㅜ내뭏 ----
                other_lp = fill_ego(partner_idx, other_lp, partner_mask_bool[~dead_agent_mask])
                world_other_lp = torch.zeros(args.dataset_size, 127).to("cuda")
                world_other_lp[(~dead_agent_mask).sum(dim=-1) == 1] = other_lp.squeeze(-1)
                other_lp_dict[future_step] = world_other_lp[torch.arange(len(intervention_idx)), matched_col].unsqueeze(-1)
            env.save_lp_pred(ego_lp_dict, other_lp_dict, ego_lp_prime_dict, matched_col)

            for world_render_idx in range(args.dataset_size):
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
        for world_render_idx in range(args.dataset_size):
            for time_step, frame in enumerate(frames[world_render_idx]):
                filename = f'scene_{world_render_idx + args.start_idx}_step_{time_step}.jpg'
                filepath = os.path.join(image_path, filename)
                imageio.imwrite(filepath, frame)
    print('finish!')
if __name__ == "__main__":
    args = parse_args()
    run(args)