from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm   # ★ 추가 필요 (파일 위쪽 import 쪽에)
import numpy as np
import torch, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from gpudrive.integrations.il.figure.near_collision import save_heatmaps, save_diff_heatmaps, to_heat, compute_overall_acc

def load_acc_torch(path, device="cuda"):
    state = torch.load(path, map_location=device)
    future_steps = state["future_steps"]
    CELLS = state["CELLS"]
    groups = state["groups"]

    from collections import defaultdict
    def make_bucket():
        return {"num": torch.zeros_like(next(iter(state["acc"].values()))[groups[0]]["num"]).to(device),
                "den": torch.zeros_like(next(iter(state["acc"].values()))[groups[0]]["den"]).to(device),
                "prob": torch.zeros_like(next(iter(state["acc"].values()))[groups[0]]["prob"]).to(device)}
    def make_group_dict():
        return {g: make_bucket() for g in groups}

    acc = defaultdict(make_group_dict)
    for fs, g_dict in state["acc"].items():
        fs = int(fs)
        for g, b in g_dict.items():
            acc[fs][g]["num"] = b["num"].to(device)
            acc[fs][g]["den"] = b["den"].to(device)
            acc[fs][g]["prob"] = b["prob"].to(device)

    return acc, future_steps, CELLS, groups
@torch.no_grad()
def save_diff_heatmaps_2x4(
    acc,
    future_steps,
    CELLS,
    outdir="./heatmaps_lp",
    group_pairs=(("veh_collision", "all"), ("off_road", "all")),
    row_labels=("Veh-Coll", "Off-Road"),
    transpose=True,
    origin="lower",
    min_count=10,
    acc_or_prob="prob",
):

    os.makedirs(outdir, exist_ok=True)

    # future_steps: 40,30,20,10
    fs_list = sorted(list(future_steps), reverse=True)  # [40,30,20,10]
    n_rows = len(group_pairs)
    n_cols = len(fs_list)

    # diffs[row][col] 
    diffs = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for r, (collision_group, non_collision_group) in enumerate(group_pairs):
        for c, fs in enumerate(fs_list):
            h_coll, cnt_coll = to_heat(acc[fs][collision_group], CELLS, acc_or_prob)
            h_non,  cnt_non  = to_heat(acc[fs][non_collision_group], CELLS, acc_or_prob)
            avg_all = compute_overall_acc(acc[fs]['all'], acc_or_prob)

            mask_valid = (cnt_coll >= min_count) & (cnt_non >= min_count)
            diff = (h_coll - h_non) / avg_all
            diff = diff.masked_fill(~mask_valid, torch.nan)
            if transpose:
                diff = diff.T
            diffs[r][c] = diff

    flat_diffs = [d for row in diffs for d in row if d is not None]
    if not flat_diffs:
        return

    stacked = torch.stack(flat_diffs)
    if torch.isfinite(stacked).any():
        mask = torch.isfinite(stacked)
        abs_max = stacked[mask].abs().max().item()
    else:
        abs_max = 1.0
    vlim = abs_max if abs_max > 0 else 1.0

    cmap = cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#9e9e9e")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        dpi=300,
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
        constrained_layout=False,
    )
    axes = np.array(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    last_im = None
    for r in range(n_rows):
        for c, fs in enumerate(fs_list):
            ax = axes[r, c]
            diff = diffs[r][c]

            im = ax.imshow(
                diff.cpu().numpy(), origin=origin,
                cmap=cmap, norm=norm, aspect="equal"
            )
            ticks = np.arange(CELLS)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])

            title = f"{fs} Step Before"
            if r == 0:
                ax.set_title(title)
            else:
                ax.set_title("")

            if c == 0 and r < len(row_labels):
                fig.text(
                0.03, 0.735 - 0.45 * r,
                row_labels[r],
                
                va="center", ha="left", rotation="vertical",
                fontfamily="Times New Roman", 
                fontsize=ax.yaxis.get_label().get_fontsize() + 2,
                fontweight='bold'
                )


            ax.set_xlabel("X Grid")
            ax.set_ylabel("Y Grid")
            last_im = im

    cbar_ax = fig.add_axes([0.9, 0.10, 0.012, 0.80])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Normalized Prob. Difference")


    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"heat_diff_2x4_{acc_or_prob}.pdf"))
    plt.close(fig)

if __name__ == '__main__':
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "font.size": 15,
        "axes.linewidth": 0.8,
        "axes.titlepad": 10,
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
    })
    WINDOW=10
    acc, future_steps, CELLS, groups = load_acc_torch(f"lp_grid_acc_exc_full_{WINDOW}.pt", device="cuda")

    save_diff_heatmaps(
        acc,
        future_steps=[10, 20, 30, 40],
        CELLS=CELLS,
        outdir=f"./heatmaps_lp_exclusive_full_{WINDOW}",
        collision_group="veh_collision",
        non_collision_group="all", 
        transpose=True,
        origin="lower",
        min_count=10,
        acc_or_prob="prob"
    )
        

    save_diff_heatmaps(
        acc,
        future_steps=[10, 20, 30, 40],
        CELLS=CELLS,
        outdir=f"./heatmaps_lp_exclusive_full_{WINDOW}",
        collision_group="off_road",
        non_collision_group="all",
        transpose=True,
        origin="lower",
        min_count=10,
        acc_or_prob="prob"
    )

    save_diff_heatmaps_2x4(
        acc,
        future_steps=[10, 20, 30, 40],
        CELLS=CELLS,
        outdir=f"./heatmaps_lp_exclusive_full_{WINDOW}",
        group_pairs=(("veh_collision", "all"),
                    ("off_road", "all")),
        row_labels=("Veh-Coll", "Off-Road"),
        transpose=True,
        origin="lower",
        min_count=10,
        acc_or_prob="prob",
    )