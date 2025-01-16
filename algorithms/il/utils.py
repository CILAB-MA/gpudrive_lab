import matplotlib.pyplot as plt
import numpy as np

def visualize_partner_obs_final(obs, partner_mask):
    """
    Final refined visualization for partner_obs.
    Args:
        partner_obs: numpy.ndarray of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        partner_mask: numpy.ndarray of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1)
    """
    partner_obs = obs[-1][6: 127 * 10 + 6].reshape(127, -1)
    kMaxAgentCount, num_features = partner_obs.shape
    assert num_features >= 6, "partner_obs feature count must be >= 6"

    # Relative positions, speeds, headings
    relative_positions = partner_obs[:, 1:3]
    speeds = partner_obs[:, 0]
    headings = partner_obs[:, 3]

    # Apply mask
    partner_mask = partner_mask.astype('bool')
    valid_mask = ~partner_mask
    relative_positions = relative_positions[valid_mask]
    speeds = speeds[valid_mask]
    headings = headings[valid_mask]
    headings *= 2 * np.pi
    # Relative position scatter plot
    fig = plt.figure(figsize=(10, 8))
    plt.quiver(
        relative_positions[:, 0],
        relative_positions[:, 1],
        np.cos(headings),
        np.sin(headings),
        angles="xy",
        scale_units="xy",
        scale=100,
        width=0.002,
        headwidth=3,
        headlength=3,
        color="red",
        label="Heading",
    )
    scatter = plt.scatter(
        relative_positions[:, 0],
        relative_positions[:, 1],
        c=speeds,
        cmap="viridis",
        alpha=0.8,
        edgecolors="k",
        s=100,  # Increased marker size
        label="Speed",
    )
    plt.colorbar(scatter, label="Speed")
    plt.xlabel("Relative X Position")
    plt.ylabel("Relative Y Position")
    plt.title("Final Refined Relative Positions of Partner Agents")
    plt.grid(True)
    plt.legend()
    return fig
