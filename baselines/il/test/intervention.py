"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging, imageio
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse, functools
from collections import defaultdict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# GPUDrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Intervention Experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/data_cut_add')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_seed_3_0523_204605.pth')
    parser.add_argument('--image-path', '-ip', type=str, default='/data/full_version/intervention')
    parser.add_argument('--make-image', '-mi', action='store_true')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=1)
    # INTERVENTION
    parser.add_argument('--intervention-idx', '-i', type=list, default=[1766, 3286, 399, 417, 1151]) # intervention partner idx
    parser.add_argument('--scene-idx', '-si', type=list, default=[0, 1, 2, 3, 4]) # intervention partner idx
    parser.add_argument('--intervention-label', '-l', type=list, default=[10] * 5) # change position label
    args = parser.parse_args()
    return args

def figure_to_numpy(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    return image

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
    TOTAL_NUM_WORLDS = 80000 if args.data_dir == "training" else 10000
    NUM_WORLDS = len(args.scene_idx)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore',
        num_stack=5

    )

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=NUM_WORLDS,
        dataset_size=TOTAL_NUM_WORLDS,
        shuffle=False,
        scene_nums=args.scene_idx,  # Specific scene numbers to intervention experiment
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=args.max_cont_agents,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )

    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    ro_attn_layers = register_all_layers_forward_hook(bc_policy.ro_attn)
    
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

    obs = env.reset(env_idx_list=args.scene_idx)
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(NUM_WORLDS)]
    intervention_label = torch.from_numpy(np.array(args.intervention_label)).to("cuda")
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
            is_world_alive = (~dead_agent_mask).sum(axis=-1).bool()
            # Get action
            context, _, _ = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
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
                ego_world = torch.zeros((NUM_WORLDS, 1)).long().to("cuda")
                ego_world_prime = torch.zeros((NUM_WORLDS, 1)).long().to("cuda")
                other_world_pred = torch.zeros((NUM_WORLDS, 127)).long().to("cuda")
                ego_world[is_world_alive] = ego_pred.unsqueeze(-1)
                ego_world_prime[is_world_alive] = ego_pred_prime.unsqueeze(-1)
                other_world_pred[is_world_alive] = other_pred
                other_lp_dict[other_head.future_step] = other_world_pred
                ego_lp_dict[ego_head.future_step] = ego_world
                ego_lp_prime_dict[ego_head.future_step] = ego_world_prime

        all_actions[~dead_agent_mask, :] = actions
        intervention_idx = torch.tensor(args.intervention_idx).long().to('cuda')

        if args.make_image:
            partner_idx = torch.full((NUM_WORLDS, 127), -2).float().to('cuda')
            partner_idx[is_world_alive] = env.partner_ids[~dead_agent_mask].clone()
            for future_step, other_lp in other_lp_dict.items():
                intervention_mask = (partner_idx == intervention_idx.unsqueeze(-1))
                intervention_match = intervention_mask.float().argmax(dim=-1)
                other_lp_dict[future_step] = other_lp[torch.arange(NUM_WORLDS), intervention_match].unsqueeze(-1)
                other_lp_dict[future_step][~is_world_alive] = -1.0
            
            setattr(env.vis, 'ego_pred_pos', ego_lp_dict)
            setattr(env.vis, 'ego_pred_prime', ego_lp_prime_dict)
            setattr(env.vis, 'other_pred', other_lp_dict)
            setattr(env.vis, 'intervention_idx', intervention_idx)

            for world_render_idx in range(NUM_WORLDS):
                frame = env.vis.plot_simulator_state(
                    env_indices=list(range(NUM_WORLDS)),
                    time_steps=[time_step]*NUM_WORLDS,
                    plot_importance_weight=False,
                    plot_linear_probing=True,
                    plot_linear_probing_label=False
                )
            for i in range(NUM_WORLDS):
                frames[i].append(frame[i])

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()

        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    
    if args.make_image:
        image_path = os.path.join(args.image_path, args.data_dir, args.model_name)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        for world_render_idx in args.scene_idx:
            world_image_path = os.path.join(image_path, f'scene_{world_render_idx}')
            if not os.path.exists(world_image_path):
                os.makedirs(world_image_path)
            for time_step, frame in enumerate(frames[world_render_idx]):
                filename = f'step_{time_step}.jpg'
                filepath = os.path.join(world_image_path, filename)
                img_array = figure_to_numpy(frame)
                imageio.imwrite(filepath, img_array)


if __name__ == "__main__":
    args = parse_args()
    run(args)