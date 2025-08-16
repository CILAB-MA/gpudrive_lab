import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append(os.getcwd())
from baselines.il.linear_probing import get_dataloader, register_all_layers_forward_hook

def evaluate_and_plot_heatmap(layers, lp_model, data_loader, device="cuda"):
    pos_correct = torch.zeros(64, device=device)
    pos_total = torch.zeros(64, device=device)

    lp_model.eval()
    with torch.no_grad():
        for batch in data_loader:
            obs, actions, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos, labels = batch
            obs = obs.to(device)
            future_pos = future_pos.to(device)
            valid_mask = valid_mask.to(device)
            future_mask = future_mask.to(device)
            partner_mask = partner_mask.to(device)
            road_mask = road_mask.to(device)
            all_masks = [partner_mask, road_mask]

            context, *_ = backbone.get_context(obs, all_masks)
            nth_layer = list(layers.keys())[-1]
            lp_input = layers[nth_layer][:, 1:128, :]
            pred_pos = lp_model(lp_input)

            future_mask = ~future_mask

            pred_label = pred_pos.argmax(dim=-1)
            correct = (pred_label == future_pos) & future_mask

            for i in range(64):
                pos_correct[i] += correct[future_pos == i].sum()
                pos_total[i] += (future_pos[future_mask] == i).sum()

    # Accuracy 히트맵
    pos_accuracy = pos_correct / (pos_total + 1e-8)
    pos_accuracy = pos_accuracy.cpu().numpy().reshape(8, 8)#.transpose(1, 0)
    
    # Count 히트맵
    pos_counts = pos_total.cpu().numpy().reshape(8, 8)#.transpose(1, 0)

    # Accuracy heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pos_accuracy, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title("Per-position Accuracy Heatmap")
    plt.xlabel("y_bin")  # <-- 바꿔줌
    plt.ylabel("x_bin")  # <-- 바꿔줌
    plt.tight_layout()
    plt.savefig("heatmap_accuracy.png")
    plt.close()

    # Count heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pos_counts, annot=True, fmt=".0f", cmap="Oranges", cbar=True)
    plt.title("Per-position Sample Count Heatmap")
    plt.xlabel("y_bin")  # <-- 바꿔줌
    plt.ylabel("x_bin")  # <-- 바꿔줌
    plt.tight_layout()
    plt.savefig("heatmap_count.png")
    plt.close()

    print("Saved: heatmap_accuracy.png & heatmap_count.png")

if __name__ == "__main__":
    from box import Box
    import yaml

    with open('baselines/il/config/lp.yaml', "r") as f:
        exp_config = Box(yaml.safe_load(f))

    eval_data_path = os.path.join('/data/full_version', 'processed/final')
    eval_data_file = "label/validation_trajectory_2500.npz"
    model_base_path = '/data/full_version/model/cov1792_clip10'
    lp_file = 'pos_early_lp_10.pth'
    lp_path = os.path.join(model_base_path, 'other_linear_prob/early_attn_s3_0630_072820_60000/seed3')

    eval_loader = get_dataloader(eval_data_path, eval_data_file, exp_config, isshuffle=False)

    backbone = torch.load(os.path.join(model_base_path, "early_attn_s3_0630_072820_60000.pth"), weights_only=False)
    lp_model = torch.load(os.path.join(lp_path, lp_file), weights_only=False)

    backbone.eval()
    lp_model.eval()
    layers = register_all_layers_forward_hook(backbone.fusion_attn)

    evaluate_and_plot_heatmap(layers, lp_model, eval_loader, device="cuda")
