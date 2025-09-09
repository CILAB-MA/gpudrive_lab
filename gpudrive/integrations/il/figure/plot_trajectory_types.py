import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

from gpudrive.env.env_torch import GPUDriveTorchEnv

import argparse


parser = argparse.ArgumentParser()
# env
parser.add_argument('--world-idx', type=int, default=0)
parser.add_argument('--draw-expert-trajectories', type=bool, default=True)
parser.add_argument('--draw-only-controllable-veh', type=bool, default=True)

# plot
parser.add_argument('--render-3d', type=bool, default=False)
parser.add_argument('--zoom-radius', type=int, default=100)
parser.add_argument('--dpi', type=int, default=300)

# save
parser.add_argument('--save-dir', type=str, default='/data/full_version/trajectory_types')
parser.add_argument('--save-name', type=str, default='trajectory_types.png')
args = parser.parse_args()

# Increase the resolution of the figure
plt.rcParams['figure.dpi'] = args.dpi  # Higher DPI for better resolution

# Configs
render_config = RenderConfig(
    render_3d=args.render_3d,
    draw_expert_trajectories=args.draw_expert_trajectories,
    draw_only_controllable_veh=args.draw_only_controllable_veh,
    obj_idx_font_size=9,
)
env_config = EnvConfig(dynamics_model="delta_local")

# Create data loader
train_loader = SceneDataLoader(
    root="/data/full_version/data/training",
    batch_size=1,
    dataset_size=args.world_idx + 1,
    sample_with_replacement=True,
    start_idx=args.world_idx
)   


env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=1,
    device="cpu",
    render_config=render_config,
    action_type="continuous" # "continuous" or "discrete"
)

_ = env.reset()

# Plot a bird's eye view of the environment
sim_state_figs = env.vis.plot_simulator_state(
    env_indices=[0],
    zoom_radius=args.zoom_radius,
    center_agent_indices=[0],
    time_steps=[0],
    plot_log_replay_trajectory=args.draw_expert_trajectories
)

os.makedirs(args.save_dir, exist_ok=True)
Image.fromarray(img_from_fig(sim_state_figs[0])).save(
                            f"{args.save_dir}/{args.save_name}",
                            dpi=(args.dpi, args.dpi)
                        )