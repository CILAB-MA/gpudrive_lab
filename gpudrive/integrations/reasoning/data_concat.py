import os
import numpy as np
from tqdm import tqdm
import gc
def run(save_path, save_name, subset_path, num_scenes, start_idx):
    def get_sorted_files(path, num_scenes, start_idx, concat_other=False):
        files = os.listdir(path)
        if not concat_other:
            files.remove('global')
            files.remove('filtered')
            files.remove('reasoning')
            files.remove('nlp')
            files = [f for f in files if int(f.split('_')[1].split('.')[0]) < num_scenes and int(f.split('_')[1].split('.')[0]) >= start_idx]
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            files = [f for f in files if int(f.split('_')[2].split('.')[0]) < num_scenes and int(f.split('_')[2].split('.')[0]) >= start_idx and concat_other in f]
            files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        return files

    subset_datas = get_sorted_files(subset_path ,num_scenes, start_idx)
    global_datas = get_sorted_files(subset_path + '/global' ,num_scenes, start_idx, concat_other='global')
    reasoning_datas = get_sorted_files(subset_path + '/filtered/reasoning' ,num_scenes, start_idx, concat_other=
                                        'reasoning')
    nlp_datas = get_sorted_files(subset_path + '/filtered/nlp' ,num_scenes, start_idx, concat_other=
                                        'reasoning')
    obs = []
    actions = []
    dead_masks = []
    partner_masks = []
    road_masks = []
    ego_global_rots = []
    ego_global_poss = []
    
    env_qs, ego_qs, sur_qs, int_qs = [], [], [], [] 
    env_q_nlp, ego_q_nlp, sur_q_nlp, int_q_nlp = [], [], [], []
    env_pas, ego_pas, sur_pas, int_pas = [], [], [], [] 
    env_pa_nlp, ego_pa_nlp, sur_pa_nlp, int_pa_nlp = [], [], [], []
    env_nas, ego_nas, sur_nas, int_nas = [], [], [], [] 
    env_na_nlp, ego_na_nlp, sur_na_nlp, int_na_nlp = [], [], [], []
    env_masks, ego_masks, sur_masks, int_masks = [], [], [], [] 
    

    for (subset_data, global_data, reasoning_data, nlp_data) in tqdm(zip(subset_datas, global_datas, reasoning_datas, nlp_datas)):
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

        nlp_data_path = os.path.join(subset_path + '/filtered/nlp', nlp_data)
        data = np.load(nlp_data_path, allow_pickle=True)
        env_q_nlp.append(data['env_q'])
        ego_q_nlp.append(data['ego_q'])
        sur_q_nlp.append(data['sur_q'])
        int_q_nlp.append(data['int_q'])
        env_pa_nlp.append(data['env_pos_a'])
        ego_pa_nlp.append(data['ego_pos_a'])
        sur_pa_nlp.append(data['sur_pos_a'])
        int_pa_nlp.append(data['int_pos_a'])
        env_na_nlp.append(data['env_neg_a'])
        ego_na_nlp.append(data['ego_neg_a'])
        sur_na_nlp.append(data['sur_neg_a'])
        int_na_nlp.append(data['int_neg_a'])
        del data
        gc.collect()

        reasoning_data_path = os.path.join(subset_path + '/filtered/reasoning', reasoning_data)
        data = np.load(reasoning_data_path)
        env_qs.append(data['env_q'])
        ego_qs.append(data['ego_q'])
        sur_qs.append(data['sur_q'])
        int_qs.append(data['int_q'])
        env_pas.append(data['env_pos_a'])
        ego_pas.append(data['ego_pos_a'])
        sur_pas.append(data['sur_pos_a'])
        int_pas.append(data['int_pos_a'])
        env_nas.append(data['env_neg_a'])
        ego_nas.append(data['ego_neg_a'])
        sur_nas.append(data['sur_neg_a'])
        int_nas.append(data['int_neg_a'])
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
    env_nas = np.concatenate(env_nas, axis=0)
    ego_nas = np.concatenate(ego_nas, axis=0)
    sur_nas = np.concatenate(sur_nas, axis=0)
    int_nas = np.concatenate(int_nas, axis=0) 
    env_pas = np.concatenate(env_pas, axis=0)
    ego_pas = np.concatenate(ego_pas, axis=0)
    sur_pas = np.concatenate(sur_pas, axis=0)
    int_pas = np.concatenate(int_pas, axis=0) 
    env_masks = np.concatenate(env_masks, axis=0)
    ego_masks = np.concatenate(ego_masks, axis=0)
    sur_masks = np.concatenate(sur_masks, axis=0)
    int_masks = np.concatenate(int_masks, axis=0)

    env_q_nlp = np.concatenate(env_q_nlp, axis=0)
    ego_q_nlp = np.concatenate(ego_q_nlp, axis=0)
    sur_q_nlp = np.concatenate(sur_q_nlp, axis=0)
    int_q_nlp = np.concatenate(int_q_nlp, axis=0)
    env_pa_nlp = np.concatenate(env_pa_nlp, axis=0)
    ego_pa_nlp = np.concatenate(ego_pa_nlp, axis=0)
    sur_pa_nlp = np.concatenate(sur_pa_nlp, axis=0)
    int_pa_nlp = np.concatenate(int_pa_nlp, axis=0) 
    env_na_nlp = np.concatenate(env_na_nlp, axis=0)
    ego_na_nlp = np.concatenate(ego_na_nlp, axis=0)
    sur_na_nlp = np.concatenate(sur_na_nlp, axis=0)
    int_na_nlp = np.concatenate(int_na_nlp, axis=0) 
    
    print("compressing!!!")
    np.savez_compressed(os.path.join(save_path, save_name), obs=obs, actions=actions, dead_mask=dead_masks, partner_mask=partner_masks, road_mask=road_masks,  )
    np.savez_compressed(os.path.join(save_path, 'global_' + save_name), ego_global_rot=ego_global_rots,ego_global_pos=ego_global_poss )
    np.savez_compressed(os.path.join(save_path, 'reasoning_question_' + save_name), env_qs=env_qs,ego_qs=ego_qs,
                        sur_qs=sur_qs,int_qs=int_qs, env_masks=env_masks,ego_masks=ego_masks,sur_masks=sur_masks,
                        int_masks=int_masks)
    np.savez_compressed(os.path.join(save_path, 'nlp_' + save_name), env_qs=env_q_nlp,ego_qs=ego_q_nlp,
                        sur_qs=sur_q_nlp,int_qs=int_q_nlp,
                        env_pos_as=env_pa_nlp,ego_pos_as=ego_pa_nlp,sur_pos_as=sur_pa_nlp, int_pos_as=int_pa_nlp,
                        env_neg_as=env_na_nlp,ego_neg_as=ego_na_nlp,sur_neg_as=sur_na_nlp, int_neg_as=int_na_nlp,)
    np.savez_compressed(os.path.join(save_path, 'reasoning_answer_' + save_name),     
                        env_pos_as=env_pas,ego_pos_as=ego_pas,sur_pos_as=sur_pas, int_pos_as=int_pas,
                        env_neg_as=env_nas,ego_neg_as=ego_nas,sur_neg_as=sur_nas, int_neg_as=int_nas,)
    print("done!!!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=80000)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    args = parser.parse_args()  
    save_path = f"/data/full_version/reasoning/processed/final/"
    save_name = f'{args.dataset}_trajectory_{args.num_scene}.npz'
    subset_path = f'/data/full_version/reasoning/processed/{args.dataset}_subset'
    os.makedirs(save_path, exist_ok=True)
    run(save_path, save_name, subset_path, args.num_scene, args.start_idx)