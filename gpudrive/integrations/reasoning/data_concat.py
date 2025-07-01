import os
import numpy as np
from tqdm import tqdm
import gc
def run(save_path, save_name, subset_path, num_scenes, start_idx):
    def get_sorted_files(path, num_scenes, start_idx, concat_other=False):
        files = os.listdir(path)
        if not concat_other:
            files.remove('global')
            files.remove('reasoning')
            files = [f for f in files if int(f.split('_')[1].split('.')[0]) < num_scenes and int(f.split('_')[1].split('.')[0]) >= start_idx]
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            files = [f for f in files if int(f.split('_')[2].split('.')[0]) < num_scenes and int(f.split('_')[2].split('.')[0]) >= start_idx and concat_other in f]
            files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        return files

    subset_datas = get_sorted_files(subset_path ,num_scenes, start_idx)
    global_datas = get_sorted_files(subset_path + '/global' ,num_scenes, start_idx, concat_other='global')
    reasoning_datas = get_sorted_files(subset_path + '/reasoning' ,num_scenes, start_idx, concat_other=
                                        'reasoning')
    obs = []
    actions = []
    dead_masks = []
    partner_masks = []
    road_masks = []
    ego_global_rots = []
    ego_global_poss = []
    
    env_qs, ego_qs, sur_qs, int_qs = [], [], [], [] 
    env_as, ego_as, sur_as, int_as = [], [], [], [] 
    env_masks, ego_masks, sur_masks, int_masks = [], [], [], [] 

    for (subset_data, global_data, reasoning_data) in tqdm(zip(subset_datas, global_datas, reasoning_datas)):
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

        global_data_path = os.path.join(subset_path + '/global', global_data)
        data = np.load(global_data_path)
        ego_global_poss.append(data['ego_global_pos'])
        ego_global_rots.append(data['ego_global_rot'])
        del data
        gc.collect()

        reasoning_data_path = os.path.join(subset_path + '/reasoning', reasoning_data)
        data = np.load(reasoning_data_path)
        env_qs.append(data['env_q'])
        ego_qs.append(data['ego_q'])
        sur_qs.append(data['sur_q'])
        int_qs.append(data['int_q'])
        env_as.append(data['env_a'])
        ego_as.append(data['ego_a'])
        sur_as.append(data['sur_a'])
        int_as.append(data['int_a'])
        env_masks.append(data['env_mask'])
        ego_masks.append(data['ego_mask'])
        sur_masks.append(data['sur_mask'])
        int_masks.append(data['int_mask'])
        del data
        gc.collect()

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    dead_masks = np.concatenate(dead_masks, axis=0)
    partner_masks = np.concatenate(partner_masks, axis=0)
    road_masks = np.concatenate(road_masks, axis=0)
    ego_global_rots = np.concatenate(ego_global_rots, axis=0)
    ego_global_poss = np.concatenate(ego_global_poss, axis=0)
    env_qs = np.concatenate(env_qs, axis=0)
    ego_qs = np.concatenate(ego_qs, axis=0)
    sur_qs = np.concatenate(sur_qs, axis=0)
    int_qs = np.concatenate(int_qs, axis=0)
    env_as = np.concatenate(env_as, axis=0)
    ego_as = np.concatenate(ego_as, axis=0)
    sur_as = np.concatenate(sur_as, axis=0)
    int_as = np.concatenate(int_as, axis=0) 
    env_masks = np.concatenate(env_masks, axis=0)
    ego_masks = np.concatenate(ego_masks, axis=0)
    sur_masks = np.concatenate(sur_masks, axis=0)
    int_masks = np.concatenate(int_masks, axis=0)
    
    print("compressing!!!")
    np.savez_compressed(os.path.join(save_path, save_name), obs=obs, actions=actions, dead_mask=dead_masks, partner_mask=partner_masks, road_mask=road_masks,  )
    np.savez_compressed(os.path.join(save_path, 'global_' + save_name), ego_global_rot=ego_global_rots,ego_global_pos=ego_global_poss )
    np.savez_compressed(os.path.join(save_path, 'reasoning_' + save_name), env_qs=env_qs,ego_qs=ego_qs,
                        sur_qs=sur_qs,int_qs=int_qs,env_as=env_as,ego_as=ego_as,sur_as=sur_as,
                        int_as=int_as,env_masks=env_masks,ego_masks=ego_masks,sur_masks=sur_masks,
                        int_masks=int_masks,)
    print("done!!!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=10000)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='validation', choices=['training', 'validation', 'testing'],)
    args = parser.parse_args()  
    save_path = f"/data/full_version/processed/final/reasoning"
    save_name = f'{args.dataset}_trajectory_{args.num_scene}.npz'
    subset_path = f"/data/full_version/processed/reasoning_{args.dataset}_subset"
    os.makedirs(save_path, exist_ok=True)
    run(save_path, save_name, subset_path, args.num_scene, args.start_idx)