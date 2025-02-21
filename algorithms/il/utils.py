import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

import pandas as pd
import statsmodels.api as sm

def visualize_partner_obs_final(obs, partner_mask):
    """
    Final refined visualization for partner_obs.
    Args:
        obs (np.ndarray): shape (...), containing partner_obs at the last time step.
        partner_mask (np.ndarray): shape (...), containing 0, 1, 2 for each agent.
                                   2는 시각화에서 제외 (마스킹)하고,
                                   0은 빨강 화살표, 1은 파랑 화살표로 표시한다.
    """
    
    partner_obs = obs[-1][6 : 127 * 10 + 6].reshape(127, -1)
    _, num_features = partner_obs.shape
    assert num_features >= 6, "partner_obs feature count must be >= 6"

    speeds = partner_obs[:, 0]           # speed
    relative_positions = partner_obs[:, 1:3]
    headings = partner_obs[:, 3] * 2 * np.pi

    valid_mask = (partner_mask != 2)
    valid_positions = relative_positions[valid_mask]
    valid_speeds = speeds[valid_mask]
    valid_headings = headings[valid_mask]
    valid_indices = np.arange(partner_obs.shape[0])[valid_mask]

    mask_0 = (partner_mask[valid_mask] == 0)
    mask_1 = (partner_mask[valid_mask] == 1)

    fig = plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        valid_positions[:, 0],
        valid_positions[:, 1],
        c=valid_speeds,
        cmap="viridis",
        alpha=0.8,
        edgecolors="k",
        s=100,  # Increased marker size
        label="Speed",
    )
    plt.colorbar(scatter, label="Speed")

    plt.quiver(
        valid_positions[mask_0, 0],
        valid_positions[mask_0, 1],
        np.cos(valid_headings[mask_0]),
        np.sin(valid_headings[mask_0]),
        color="red",
        angles="xy",
        scale_units="xy",
        scale=100,
        width=0.002,
        headwidth=3,
        headlength=3,
        label="Moving",
    )
    plt.quiver(
        valid_positions[mask_1, 0],
        valid_positions[mask_1, 1],
        np.cos(valid_headings[mask_1]),
        np.sin(valid_headings[mask_1]),
        color="blue",
        angles="xy",
        scale_units="xy",
        scale=100,
        width=0.002,
        headwidth=3,
        headlength=3,
        label="Static",
    )

    for i, (x, y) in enumerate(valid_positions):
        plt.text(
            x, y, str(valid_indices[i]),
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
        )

    plt.xlabel("Relative X Position")
    plt.ylabel("Relative Y Position")
    plt.title("Final Refined Relative Positions of Partner Agents")
    plt.grid(True)
    plt.legend()
    return fig, valid_indices

def visualize_embedding(
    others_tsne,
    other_distance,
    other_speeds,
    tsne_indices,
    tsne_data_mask,
    tsne_partner_mask,
    num_scenes=10,
):
    filtered_tsne_mask = tsne_data_mask[(~tsne_partner_mask).cpu().numpy()]

    other_tsne_0 = others_tsne[: len(tsne_indices)]
    # other_dist_0 = other_distance[: len(tsne_indices)]

    fig, (ax_0, ax_all, ax_scene, ax_speed, ax_dist) = plt.subplots(1, 5, figsize=(30, 6))

    tsne_first = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    emb_tsne_0 = tsne_first.fit_transform(other_tsne_0)
    x_0, y_0 = emb_tsne_0[:, 0], emb_tsne_0[:, 1]

    mask_0 = filtered_tsne_mask[: len(tsne_indices)]

    face_colors_0 = []
    for m in mask_0:
        if m == 0:  # Moving
            face_colors_0.append((1.0, 0.0, 0.0, 1.0))
        else:       # Static
            face_colors_0.append((0.0, 0.0, 1.0, 1.0))

    ax_0.scatter(x_0, y_0, facecolors=face_colors_0, edgecolors="none", s=100)

    for j in range(len(tsne_indices)):
        ax_0.text(
            x_0[j], y_0[j], str(tsne_indices[j]),
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.1, edgecolor="none")
        )

    legend_handles_0 = [
        mpatches.Patch(color='red', label='Moving'),
        mpatches.Patch(color='blue', label='Static'),
    ]
    ax_0.legend(handles=legend_handles_0)
    ax_0.set_title("TSNE Visualization (First Scene)")

    tsne_all = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    emb_tsne_all = tsne_all.fit_transform(others_tsne)
    x_all, y_all = emb_tsne_all[:, 0], emb_tsne_all[:, 1]

    face_colors_all = []
    for m in filtered_tsne_mask:
        if m == 0:  # Moving
            face_colors_all.append((1.0, 0.0, 0.0, 1.0))
        else:       # Static
            face_colors_all.append((0.0, 0.0, 1.0, 1.0))

    ax_all.scatter(x_all, y_all, facecolors=face_colors_all, edgecolors="none", s=50)

    legend_handles_all = [
        mpatches.Patch(color='red', label='Moving'),
        mpatches.Patch(color='blue', label='Static'),
    ]
    ax_all.legend(handles=legend_handles_all)
    ax_all.set_title("TSNE Visualization (All Scenes)")

    scene_indices_2d = (
        torch.arange(num_scenes)
        .unsqueeze(1)
        .expand(num_scenes, 127)
        .to(tsne_partner_mask.device)
    )
    scene_indices_masked = scene_indices_2d[~tsne_partner_mask]

    unique_scenes = torch.unique(scene_indices_masked).cpu().numpy()
    cmap_scene = plt.cm.get_cmap('tab10', len(unique_scenes))

    face_colors_scene = [cmap_scene(idx.item()) for idx in scene_indices_masked]

    ax_scene.scatter(x_all, y_all, facecolors=face_colors_scene, edgecolors="none", s=50)

    legend_handles_scene = []
    for s in unique_scenes:
        color_i = cmap_scene(s)
        legend_handles_scene.append(
            mpatches.Patch(color=color_i, label=f'Scene {s}')
        )
    ax_scene.legend(handles=legend_handles_scene)
    ax_scene.set_title("TSNE Visualization (Colored by Scene)")

    sc_speed = ax_speed.scatter(
        x_all,
        y_all,
        c=other_speeds,
        cmap="viridis",
        edgecolors="none",
        s=50
    )
    cbar_speed = fig.colorbar(sc_speed, ax=ax_speed)
    cbar_speed.set_label("Speed")
    ax_speed.set_title("TSNE Visualization (By Speed)")

    sc_dist = ax_dist.scatter(
        x_all,
        y_all,
        c=other_distance,
        cmap="viridis",
        edgecolors="none",
        s=50
    )
    cbar_dist = fig.colorbar(sc_dist, ax=ax_dist)
    cbar_dist.set_label("Distance")
    ax_dist.set_title("TSNE Visualization (By Distance)")

    plt.tight_layout()
    return fig

def visualize_tsne_with_weights(
    other_tsne,
    other_weights,
    tsne_data_mask,
    tsne_partner_mask
):
    filtered_tsne_mask = tsne_data_mask[(~tsne_partner_mask).cpu().numpy()]
    tsne_model = TSNE(
        n_components=2, perplexity=30, learning_rate='auto',
        init='random', random_state=42
    )
    aux_task = ['action', 'pos', 'heading', 'speed']
    emb_tsne = tsne_model.fit_transform(other_tsne)
    x, y = emb_tsne[:, 0], emb_tsne[:, 1]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    for i in range(4):
        sc = axes[i].scatter(
            x,
            y,
            c=other_weights[:, i],
            cmap='viridis',
            edgecolors='none',
            s=50
        )
        cbar = fig.colorbar(sc, ax=axes[i])
        cbar.set_label(f'Weight {aux_task[i]}')
        axes[i].set_title(f'TSNE Visualization (Weight {aux_task[i]})')

    plt.tight_layout()
    return fig

def compute_correlation_scatter(dist, coll, loss):
    data = np.vstack([dist, coll, loss])
    corr_matrix = np.corrcoef(data)
    df_corr = pd.DataFrame(corr_matrix, index=['Current Distance', 'Distance Difference', 'Loss'], 
                           columns=['Current Distance', 'Distance Difference', 'Loss'])
    x = np.column_stack((dist, coll))
    x = sm.add_constant(x)
    model = sm.OLS(loss, x).fit()
    print(model.summary())
    r_squared = model.rsquared
    multiple_correlation = np.sqrt(r_squared)
    print(f"Multiple Correlation Coefficient (R): {multiple_correlation:.4f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(dist, coll, c=loss, cmap='viridis', edgecolor='none', alpha=0.7)
    plt.colorbar(scatter, label="Loss Value")
    ax.set_xlabel("Current Distance")
    ax.set_ylabel("Distance Difference")
    ax.set_title("Evaluation of Collision Risk")
    ax.grid(True)

    return df_corr, fig