import os
import numpy as np
from tqdm import tqdm
import gc
import pandas as pd

def run(save_path, save_name, subset_path, num_scenes, start_idx):
    def get_sorted_files(path, num_scenes, start_idx, concat_other=False):
        files = os.listdir(path)
        if not concat_other:
            files.remove('global')
            files.remove('label')
            files = [f for f in files if int(f.split('_')[1].split('.')[0]) < num_scenes and int(f.split('_')[1].split('.')[0]) >= start_idx]
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            files = [f for f in files if int(f.split('_')[2].split('.')[0]) < num_scenes and int(f.split('_')[2].split('.')[0]) >= start_idx and concat_other in f]
            files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        return files

    subset_datas = get_sorted_files(subset_path ,num_scenes, start_idx)
    global_datas = get_sorted_files(subset_path + '/global' ,num_scenes, start_idx, concat_other='global')
    label_datas = get_sorted_files(subset_path + '/label' ,num_scenes, start_idx, concat_other='label')
    obs = []
    actions = []
    dead_masks = []
    partner_masks = []
    road_masks = []
    ego_global_rots = []
    ego_global_poss = []
    ego_labels = []
    partner_labels = []
    for idx, (subset_data, global_data, label_data) in tqdm(enumerate(zip(subset_datas, global_datas, label_datas))):
        subset_data_path = os.path.join(subset_path, subset_data)
        data = np.load(subset_data_path)
        obs.append(data['obs'])
        actions.append(data['actions'])
        dead_masks.append(data['dead_mask'])
        partner_masks.append(data['partner_mask'])
        road_masks.append(data['road_mask'])
        del data
        gc.collect()

        global_data_path = os.path.join(subset_path + '/global', global_data)
        data = np.load(global_data_path)
        ego_global_poss.append(data['ego_global_pos'])
        ego_global_rots.append(data['ego_global_rot'])
        del data
        gc.collect()

        label_data_path = os.path.join(subset_path + '/label', label_data)
        data = np.load(label_data_path)
        partner_labels.append(data['partner_label'])
        ego_labels.append(data['ego_label'])
        del data
        gc.collect()

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    dead_masks = np.concatenate(dead_masks, axis=0)
    partner_masks = np.concatenate(partner_masks, axis=0)
    road_masks = np.concatenate(road_masks, axis=0)
    ego_labels = np.concatenate(ego_labels, axis=0)
    partner_labels = np.concatenate(partner_labels, axis=0)
    ego_global_rots = np.concatenate(ego_global_rots, axis=0)
    ego_global_poss = np.concatenate(ego_global_poss, axis=0)
    print("compressing!!!")
    np.savez_compressed(os.path.join(save_path, save_name), obs=obs, actions=actions, 
                        dead_mask=dead_masks, partner_mask=partner_masks, 
                        road_mask=road_masks, partner_labels=partner_labels, ego_labels=ego_labels)
    # np.savez_compressed(os.path.join(save_path, 'global_' + save_name), ego_global_rot=ego_global_rots,ego_global_pos=ego_global_poss )
    print("done!!!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=2500)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='validation', choices=['training', 'validation', 'testing'],)
    args = parser.parse_args()
    save_path = "/data/full_version/processed/final"
    save_name = f'label/{args.dataset}_trajectory_{args.num_scene}.npz'
    subset_path = f"/data/full_version/processed/{args.dataset}_subset_v2"
    os.makedirs(save_path, exist_ok=True)
    run(save_path, save_name, subset_path, args.num_scene, args.start_idx)