import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import os, sys

from baselines.il.config import NetworkConfig, ExperimentConfig, EnvConfig
from baselines.il.dataloader import ExpertDataset
from algorithms.il.model.bc import *
from algorithms.il import MODELS, LOSS

def get_dataloader():
    with np.load("/data/tom/test_trajectory_200.npz") as npz:
        expert_obs = [npz['obs']]
        expert_actions = [npz['actions']]
        expert_masks = [npz['dead_mask']] if 'dead_mask' in npz.keys() else []
        other_info = [npz['other_info']] if 'other_info' in npz.keys() else []
        road_mask = [npz['road_mask']] if 'road_mask' in npz.keys() else []

    expert_obs = expert_obs[0][:256,...]
    expert_actions = expert_actions[0][:256,...]
    expert_masks = expert_masks[0][:256,...] if len(expert_masks) > 0 else None
    other_info = other_info[0][:256,...] if len(other_info) > 0 else None
    road_mask = road_mask[0][:256,...] if len(road_mask) > 0 else None

    data_loader = DataLoader(
        ExpertDataset(
            expert_obs, expert_actions, expert_masks,
            other_info=other_info, road_mask=road_mask,
            rollout_len=10, pred_len=5
        ),
        batch_size=256,
        shuffle=False,
        num_workers=int(os.cpu_count() / 2),
        pin_memory=True
    )
    del expert_obs, expert_actions, expert_masks, other_info, road_mask
    return data_loader

if __name__ == "__main__":
    torch.cuda.empty_cache()
    net_config = NetworkConfig()
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
        ).to("cuda"),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to("cuda"),
        dyaw=torch.round(
            torch.linspace(-np.pi, np.pi, 100), decimals=3
        ).to("cuda"),
    )

    dataloader = get_dataloader()
    model = MODELS["wayformer"](env_config, net_config, "gmm", 1).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # TensorBoard 저장 경로
        record_shapes=True,
        with_stack=True
    ) as prof:
        for i, batch in enumerate(dataloader):
            obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch
            
            obs = obs.to("cuda")
            expert_action = expert_action.to("cuda")
            masks = masks.to("cuda")
            ego_masks = ego_masks.to("cuda")
            partner_masks = partner_masks.to("cuda")
            road_masks = road_masks.to("cuda")
            all_masks = [masks, ego_masks, partner_masks, road_masks]

            with record_function("forward_pass(get_context)"):
                context = model.get_embedded_obs(obs, all_masks[1:])
            with record_function("forward_pass(get_action)"):
                pred_actions = model.get_action(context, deterministic=True)
            with record_function("forward_pass"):
                pred_actions = model(obs, all_masks[1:], deterministic=True)
            with record_function("forward_pass(get_loss)"):
                loss = LOSS["gmm"](model, context, expert_action, all_masks)
            with record_function("backward_pass"):
                loss.backward()
            with record_function("optimization_step"):
                optimizer.step()
                optimizer.zero_grad()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))