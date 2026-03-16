import os
import numpy as np
from tqdm import tqdm
import gc
import random


def extract_scene_id(filename):
    return int(filename.split("_")[1].split(".")[0])


def compute_scene_unit(scene_ids):
    diffs = [scene_ids[i+1] - scene_ids[i] for i in range(len(scene_ids)-1)]
    return min(diffs)


def run(save_path, save_name, subset_path, num_scenes, seed=42):
    all_files = [f for f in os.listdir(subset_path) if f.endswith(".npz") and f.startswith("trajectory_")]
    scene_ids = sorted([extract_scene_id(f) for f in all_files])

    if len(scene_ids) < 2:
        raise ValueError("Num of scene is less than 2.")

    scene_unit = compute_scene_unit(scene_ids)
    print(f"[INFO] scene_unit = {scene_unit}")

    need_files = (num_scenes + scene_unit - 1) // scene_unit

    if need_files > len(scene_ids):
        raise ValueError("Num scene is less than requeseted numbers.")

    random.seed(seed)
    chosen_ids = random.sample(scene_ids, need_files)
    chosen_ids = sorted(chosen_ids)

    chosen_files = [f"trajectory_{sid}.npz" for sid in chosen_ids]

    obs_list = []
    actions_list = []
    dead_masks_list = []
    partner_masks_list = []
    road_masks_list = []
    ego_global_rots_list = []
    ego_global_poss_list = []
    partner_labels_list = []
    ego_labels_list = []

    label_path = os.path.join(subset_path, "label")
    has_labels = os.path.isdir(label_path)

    total_loaded = 0

    for filename in tqdm(chosen_files):
        file_path = os.path.join(subset_path, filename)
        data = np.load(file_path)

        obs_list.append(data["obs"])
        actions_list.append(data["actions"])
        dead_masks_list.append(data["dead_mask"])
        partner_masks_list.append(data["partner_mask"])
        road_masks_list.append(data["road_mask"])

        scene_count = data["obs"].shape[0]
        total_loaded += scene_count

        del data
        gc.collect()

        global_file = f"global_{filename}"
        global_path = os.path.join(subset_path, "global", global_file)

        g = np.load(global_path)
        ego_global_rots_list.append(g["ego_global_rot"])
        ego_global_poss_list.append(g["ego_global_pos"])
        del g
        gc.collect()

        if has_labels:
            label_filename = f"label_{filename}"  # label_trajectory_{sid}.npz
            label_file_path = os.path.join(label_path, label_filename)
            if os.path.exists(label_file_path):
                lbl = np.load(label_file_path)
                partner_labels_list.append(lbl["partner_label"])
                ego_labels_list.append(lbl["ego_label"])
                del lbl
                gc.collect()
            else:
                print(f"[WARN] missing {label_file_path}, skipping all labels")
                partner_labels_list.clear()
                ego_labels_list.clear()
                has_labels = False

    obs = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    dead_masks = np.concatenate(dead_masks_list, axis=0)
    partner_masks = np.concatenate(partner_masks_list, axis=0)
    road_masks = np.concatenate(road_masks_list, axis=0)

    ego_global_rots = np.concatenate(ego_global_rots_list, axis=0)
    ego_global_poss = np.concatenate(ego_global_poss_list, axis=0)

    save_dict = dict(
        obs=obs,
        actions=actions,
        dead_mask=dead_masks,
        partner_mask=partner_masks, 
        road_mask=road_masks,
    )
    if partner_labels_list and ego_labels_list:
        partner_labels = np.concatenate(partner_labels_list, axis=0)
        ego_labels = np.concatenate(ego_labels_list, axis=0)
        save_dict["partner_labels"] = partner_labels
        save_dict["ego_labels"] = ego_labels
        print(f"[INFO] concat labels: partner {partner_labels.shape}, ego {ego_labels.shape}")

    print("[INFO] compressing & saving...")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "global"), exist_ok=True)

    np.savez_compressed(os.path.join(save_path, save_name), **save_dict)

    np.savez_compressed(
        os.path.join(save_path, "global", f"global_{save_name}"),
        ego_global_rot=ego_global_rots,
        ego_global_pos=ego_global_poss,
    )

    print("[INFO] done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=100,
                        help='Scene key for paths: save_path/subset_path use scene_{scene} (e.g. 100 -> scene_100)')
    parser.add_argument('--num-scene', type=int, default=2500,
                        help='Number of scenes to sample into the concatenated file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='validation', choices=['training', 'validation', 'testing'])
    args = parser.parse_args()

    base = "/data/after_cvpr/linear_probe_data"
    scene_key = f"scene_{args.scene}"
    save_path = f"{base}/{scene_key}"
    save_name = f"{args.dataset}_trajectory_{args.num_scene}.npz"
    subset_path = f"{base}/{scene_key}/{args.dataset}_rl_data"

    run(save_path, save_name, subset_path, args.num_scene, seed=args.seed)
