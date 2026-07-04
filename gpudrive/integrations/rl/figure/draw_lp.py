import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
# ---------------------- 설정 ----------------------
data_sources = [
    {
        "label": "IL",
        "path": "Z:\\kevinjeon\\nocturne\\full_version\\linear_probing_final",
        "scene_nums": [100, 1000, 10000],
    },
    {
        "label": "RL",
        "path": "Z:\\kevinjeon\\nocturne\\after_cvpr\\linear_probing_final",
        "scene_nums": [100, 1000, 10000],
    },
]
exp   = "other"   # 'ego'면 블루/오렌지, 그 외면 보라/그린
top_row_future_steps = [10, 20, 30, 40]
bottom_row_fs = 10
scene_labels_k = []
models = ["baseline", "lp", "early_lp", "final_lp"]
action_metrics = [
    ("eval/normal_acc",   "Uncategorized"),
    ("eval/turn_acc",     "Turn"),
    ("eval/straight_acc", "Straight"),
    ("eval/retreat_acc",  "Reverse"),
]


# -------- 팔레트 스위치 (exp에 따라 자동 전환) --------
model_order = list(models)
def _shade_series(base_hex):
    r, g, b = to_rgb(base_hex)
    # baseline(연한) / lp(진한)
    def mix(t, target=(1, 1, 1)):  # t=0 원색, t=1 흰색
        return ((1-t)*r + t*target[0], (1-t)*g + t*target[1], (1-t)*b + t*target[2])
    light  = mix(0.75)           # 연하게
    dark   = mix(0.00)           # 진한(원색)
    return {"baseline": light, "lp": dark}

def pick_palettes_set2(exp_name: str):
    set2 = sns.color_palette("Set2", n_colors=8)  # [0..7]
    exp_name = str(exp_name).lower()

    if exp_name == "other":
        base_f1  = set2[2]  # purple-ish
        base_acc = set2[0]  # green-ish
    else:  # ego
        base_f1  = set2[3]  # pink/magenta
        base_acc = set2[1]  # orange

    model_colors_f1  = _shade_series(base_f1)
    model_colors_acc = _shade_series(base_acc)
    return model_colors_f1, model_colors_acc

# 기존 pick_palettes(...) 대신 사용
model_colors_f1, model_colors_acc = pick_palettes_set2(exp)
source_plot_style = {
    "IL": {
        "line_color": "#1f77b4",  # blue
        "fill_color": "#1f77b4",
        "line_style": "-",
        "marker": "o",
    },
    "RL": {
        "line_color": "#d62728",  # red
        "fill_color": "#d62728",
        "line_style": "--",
        "marker": "s",
    },
}

# ---------------------- 데이터 로드 ----------------------
source_data = {}
for src in data_sources:
    src_label = src["label"]
    src_path = src["path"]
    src_scenes = src["scene_nums"]
    src_data = {}
    loaded_scenes = []
    for scene_num in src_scenes:
        candidate_files = [
            f"lp{scene_num}_v2.csv",
            f"lp{scene_num}.csv",
            f"rl_rl_{scene_num}.csv",
            
        ]
        fp = None
        for file in candidate_files:
            candidate_fp = os.path.join(src_path, file)
            if os.path.exists(candidate_fp):
                fp = candidate_fp
                break
        if fp is None:
            print(
                f"[WARN] file not found for scene {scene_num} in {src['label']}: {candidate_files}"
            )
            continue
        df = pd.read_csv(fp)
        original_n = len(df)
        if "experiment" in df.columns:
            filtered = df[df["experiment"].astype(str).str.lower() == exp]
            if len(filtered) == 0 and original_n > 0:
                print(
                    f"[WARN] experiment='{exp}' produced empty data for {fp}. Using unfiltered rows."
                )
            else:
                df = filtered
        src_data[str(scene_num)] = df
        loaded_scenes.append(str(scene_num))
    source_data[src_label] = {
        "path": src_path,
        "scene_nums": loaded_scenes,
        "data": src_data,
    }

all_scene_ints = sorted(
    {
        int(scene)
        for src in source_data.values()
        for scene in src["scene_nums"]
    }
)
scene_labels_k = [f"{s / 1000:g}k" for s in all_scene_ints]
primary_source = data_sources[0]["label"]
# ---------------------- 통계 유틸 ----------------------
def agg_seed_mean_std(df, metric, model_name, future_step):
    g = df[(df["model"] == model_name) & (df["future_step"] == future_step)]
    if g.empty or metric not in g.columns:
        return np.nan, np.nan
    by_seed = g.groupby("seed")[metric].mean()
    return by_seed.mean(), by_seed.std(ddof=1)


def lp_proxy_mean_std(df, metric, future_step):
    """Return LP metric in a model-schema-aware way.
    - RL schema: [baseline, lp] => use lp
    - IL schema: [baseline, early_lp, final_lp] => use max(early_lp, final_lp)
    """
    lp_mu, lp_sd = agg_seed_mean_std(df, metric, "lp", future_step)
    if not np.isnan(lp_mu):
        return lp_mu, lp_sd

    e_mu, e_sd = agg_seed_mean_std(df, metric, "early_lp", future_step)
    f_mu, f_sd = agg_seed_mean_std(df, metric, "final_lp", future_step)
    if np.isnan(e_mu) and np.isnan(f_mu):
        return np.nan, np.nan
    if (not np.isnan(e_mu)) and (np.isnan(f_mu) or e_mu >= f_mu):
        return e_mu, e_sd
    return f_mu, f_sd

# ---------------------- y-limit 산출 ----------------------
f1_diff_vals_collect = []
for fs in top_row_future_steps:
    for src in source_data.values():
        for scene in src["scene_nums"]:
            b_mu, b_sd = agg_seed_mean_std(
                src["data"][scene], "eval/pos_f1_macro", "baseline", fs
            )
            lp_mu, lp_sd = lp_proxy_mean_std(
                src["data"][scene], "eval/pos_f1_macro", fs
            )
            diff_mu = lp_mu - b_mu
            diff_sd = np.hypot(lp_sd, b_sd)
            f1_diff_vals_collect.extend([diff_mu - diff_sd, diff_mu + diff_sd])
f1_diff_vals = np.array(f1_diff_vals_collect, dtype=float)
if np.isfinite(f1_diff_vals).any():
    diff_max = np.nanmax(f1_diff_vals)
    f1_diff_ylim = (0.0, diff_max + 0.02)
else:
    f1_diff_ylim = (0.0, 0.1)

acc_diff_vals_collect = []
for met, _name in action_metrics:
    for src in source_data.values():
        for scene in src["scene_nums"]:
            b_mu, b_sd = agg_seed_mean_std(
                src["data"][scene], met, "baseline", bottom_row_fs
            )
            lp_mu, lp_sd = lp_proxy_mean_std(src["data"][scene], met, bottom_row_fs)
            diff_mu = lp_mu - b_mu
            diff_sd = np.hypot(lp_sd, b_sd)
            acc_diff_vals_collect.extend([diff_mu - diff_sd, diff_mu + diff_sd])

acc_diff_vals = np.array(acc_diff_vals_collect, dtype=float)
if np.isfinite(acc_diff_vals).any():
    acc_max = np.nanmax(acc_diff_vals)
    acc_diff_ylim = (0.0, acc_max + 0.02)
else:
    acc_diff_ylim = (0.0, 0.1)

# ---------------------- 플로팅 ----------------------
plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 25, "axes.labelsize": 19,
    "xtick.labelsize": 23, "ytick.labelsize": 23, "legend.fontsize": 18,
    "text.usetex": True,
    'font.family': 'Times New Roman',
    # "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans"],
    "mathtext.fontset": "stixsans",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Times"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})
fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharey="row")

# Top row: F1 Macro difference (LP - Raw)
for c, fs in enumerate(top_row_future_steps):
    ax = axes[0, c]
    for src_idx, src in enumerate(data_sources):
        src_label = src["label"]
        src_bundle = source_data[src_label]
        src_scene_ints = [int(s) for s in src_bundle["scene_nums"]]
        if len(src_scene_ints) == 0:
            continue
        lp_minus_raw = []
        lp_minus_raw_std = []
        for scene in src_bundle["scene_nums"]:
            b_mu, b_sd = agg_seed_mean_std(
                src_bundle["data"][scene], "eval/pos_f1_macro", "baseline", fs
            )
            lp_mu, lp_sd = lp_proxy_mean_std(
                src_bundle["data"][scene], "eval/pos_f1_macro", fs
            )
            lp_minus_raw.append(lp_mu - b_mu)
            lp_minus_raw_std.append(np.hypot(lp_sd, b_sd))

        y = np.array(lp_minus_raw, dtype=float)
        y_std = np.array(lp_minus_raw_std, dtype=float)
        valid_mask = np.isfinite(y)
        if not np.any(valid_mask):
            print(f"[WARN] no finite points to plot: source={src_label}, future_step={fs}")
            continue
        style = source_plot_style.get(
            src_label,
            {
                "line_color": "#2ca02c",
                "fill_color": "#2ca02c",
                "line_style": "-.",
                "marker": "D",
            },
        )
        ax.plot(
            np.array(src_scene_ints)[valid_mask],
            y[valid_mask],
            marker=style["marker"],
            linewidth=2.6,
            markersize=7,
            linestyle=style["line_style"],
            color=style["line_color"],
            label=f"{src_label}: LP - Raw" if c == 0 else None,
        )
        ax.fill_between(
            np.array(src_scene_ints)[valid_mask],
            (y - y_std)[valid_mask],
            (y + y_std)[valid_mask],
            color=style["fill_color"],
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylim(*f1_diff_ylim)
    ax.set_xlabel("Number of Scenes", fontsize=22, labelpad=8)
    ax.set_title(f"F1 Macro Diff (Future Step: {fs})", fontsize=24)
    ax.set_xticks(all_scene_ints)
    ax.set_xticklabels(scene_labels_k, rotation=30, ha="center")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if c == 0:
        ax.set_ylabel("F1 Macro (LP - Raw)", fontsize=24)

# Bottom row: Accuracy difference (LP - Raw), fs=bottom_row_fs
for c, (met, name) in enumerate(action_metrics):
    ax = axes[1, c]
    for src_idx, src in enumerate(data_sources):
        src_label = src["label"]
        src_bundle = source_data[src_label]
        src_scene_ints = [int(s) for s in src_bundle["scene_nums"]]
        if len(src_scene_ints) == 0:
            continue

        lp_minus_raw = []
        lp_minus_raw_std = []
        for scene in src_bundle["scene_nums"]:
            b_mu, b_sd = agg_seed_mean_std(
                src_bundle["data"][scene], met, "baseline", bottom_row_fs
            )
            lp_mu, lp_sd = lp_proxy_mean_std(
                src_bundle["data"][scene], met, bottom_row_fs
            )
            lp_minus_raw.append(lp_mu - b_mu)
            lp_minus_raw_std.append(np.hypot(lp_sd, b_sd))

        y = np.array(lp_minus_raw, dtype=float)
        y_std = np.array(lp_minus_raw_std, dtype=float)
        valid_mask = np.isfinite(y)
        if not np.any(valid_mask):
            print(f"[WARN] no finite points to plot: source={src_label}, metric={met}")
            continue

        style = source_plot_style.get(
            src_label,
            {
                "line_color": "#2ca02c",
                "fill_color": "#2ca02c",
                "line_style": "-.",
                "marker": "D",
            },
        )
        x_valid = np.array(src_scene_ints)[valid_mask]
        y_valid = y[valid_mask]
        y_std_valid = y_std[valid_mask]
        ax.plot(
            x_valid,
            y_valid,
            marker=style["marker"],
            linewidth=2.6,
            markersize=7,
            linestyle=style["line_style"],
            color=style["line_color"],
            label=None,
        )
        ax.fill_between(
            x_valid,
            y_valid - y_std_valid,
            y_valid + y_std_valid,
            color=style["fill_color"],
            alpha=0.14,
            linewidth=0,
        )

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylim(*acc_diff_ylim)
    ax.set_xlabel("Number of Scenes", fontsize=22, labelpad=8)
    ax.set_title(f"{name} Accuracy Diff (fs={bottom_row_fs})", fontsize=24)
    ax.set_xticks(all_scene_ints)
    ax.set_xticklabels(scene_labels_k, rotation=30, ha="center")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if c == 0:
        ax.set_ylabel("Accuracy (LP - Raw)", fontsize=24)

legend_handles = []
legend_labels = []
for ax in fig.axes:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label and (label not in legend_labels):
            legend_handles.append(handle)
            legend_labels.append(label)
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=max(1, len(legend_labels)),
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
    fontsize=20,
)

# fig.suptitle(
#     f"Linear Probing ({exp.capitalize()}) — Top: F1 score across future step, Bottom: Type accuracy",
#     fontsize=30, y=1.06
# )
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])
plt.subplots_adjust(bottom=0.12, hspace=0.5)
fig_num = 8 if exp == 'ego' else 9
# plt.savefig(f"{fig_num}_{exp}_linear_probing_paper_palette.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"{fig_num}_{exp}_linear_probing_paper_palette.svg", bbox_inches="tight")
plt.savefig(f"{fig_num}_{exp}_linear_probing_paper_palette.pdf", bbox_inches="tight")
plt.show()

def _mean_std_metric(scene_df: pd.DataFrame, metric: str, model: str, fs: int):
    mu, sd = agg_seed_mean_std(scene_df, metric, model, fs)
    mu = float(mu) if np.isfinite(mu) else np.nan
    sd = float(sd) if np.isfinite(sd) else np.nan
    return mu, sd

print("\n=== (A) F1 Macro: LP - raw-input ===")
for fs in top_row_future_steps:
    rows = []
    for scene in source_data[primary_source]["scene_nums"]:
        df_scene = source_data[primary_source]["data"][scene]
        met = "eval/pos_f1_macro"

        lp_mu, lp_sd = lp_proxy_mean_std(df_scene, met, fs)
        b_mu, b_sd = _mean_std_metric(df_scene, met, "baseline", fs)

        lp_raw_mu = lp_mu - b_mu
        lp_raw_sd = np.hypot(lp_sd, b_sd)

        rows.append({
            "Scene": scene,
            "lp-raw_mean": lp_raw_mu,
            "lp-raw_std":  lp_raw_sd,
            "F1(lp)_mean": lp_mu, "F1(lp)_std": lp_sd,
            "F1(raw)_mean":   b_mu, "F1(raw)_std":   b_sd,
        })
    f1_df = pd.DataFrame(rows).set_index("Scene")
    print(f"\n[Future Step = {fs}]")
    print(f1_df.round(6))
    print("Avg over scenes:", f1_df.mean(numeric_only=True).round(6).to_dict())

print("\n=== (B) Accuracy@future_step={}: LP - raw-input ===".format(bottom_row_fs))
for met, name in action_metrics:
    rows = []
    for scene in source_data[primary_source]["scene_nums"]:
        df_scene = source_data[primary_source]["data"][scene]
        lp_mu, lp_sd = lp_proxy_mean_std(df_scene, met, bottom_row_fs)
        b_mu, b_sd = _mean_std_metric(df_scene, met, "baseline", bottom_row_fs)

        lp_raw_mu = lp_mu - b_mu
        lp_raw_sd = np.hypot(lp_sd, b_sd)

        rows.append({
            "Scene": scene,
            "lp-raw_mean": lp_raw_mu,
            "lp-raw_std":  lp_raw_sd,
            f"{name}(lp)_mean": lp_mu, f"{name}(lp)_std": lp_sd,
            f"{name}(raw)_mean":   b_mu, f"{name}(raw)_std":   b_sd,
        })
    acc_df = pd.DataFrame(rows).set_index("Scene")
    print(f"\n[{name} Accuracy]")
    print(acc_df.round(6))
    print("Avg over scenes:", acc_df.mean(numeric_only=True).round(6).to_dict())
