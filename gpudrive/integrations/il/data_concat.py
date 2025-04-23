import os
import numpy as np
from tqdm import tqdm
import gc
def run(save_path, save_name, subset_path, num_scenes, start_idx):
    def get_sorted_files(path, num_scenes, start_idx):
        files = os.listdir(path)
        files = [f for f in files if int(f.split('_')[1].split('.')[0]) < num_scenes and int(f.split('_')[1].split('.')[0]) >= start_idx and "temp" not in f]
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        print(files)
        return files

    subset_datas = get_sorted_files(subset_path ,num_scenes, start_idx)
    obs = []
    actions = []
    dead_masks = []
    partner_masks = []
    road_masks = []
    other_infos = []
    
    for subset_data in tqdm(subset_datas):
        subset_data_path = os.path.join(subset_path, subset_data)
        data = np.load(subset_data_path)
        obs.append(data['obs'])
        print(f"{subset_data}, vehicle : {data['obs'].shape[0]}")
        actions.append(data['actions'])
        dead_masks.append(data['dead_mask'])
        partner_masks.append(data['partner_mask'])
        road_masks.append(data['road_mask'])
        del data
        gc.collect()

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    dead_masks = np.concatenate(dead_masks, axis=0)
    partner_masks = np.concatenate(partner_masks, axis=0)
    road_masks = np.concatenate(road_masks, axis=0)
    print("compressing!!!")
    # np.savez_compressed(os.path.join(save_path, save_name), obs=obs, actions=actions, dead_mask=dead_masks, partner_mask=partner_masks, road_mask=road_masks, other_info=other_infos, )
    print("done!!!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=1000)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    args = parser.parse_args()
    start_idx = 0
    save_path = "/data/full_version/processed/final"
    save_name = f'{args.dataset}_trajectory_{args.num_scene}.npz'
    subset_path = f"/data/full_version/processed/{args.dataset}_subset"

    run(save_path, save_name, subset_path, num_scenes, start_idx)