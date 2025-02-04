import matplotlib.pyplot as plt
import numpy as np

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
    kMaxAgentCount, num_features = partner_obs.shape
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
