import json
from tqdm import tqdm
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    env_qs = []
    ego_qs = []
    sur_qs = []
    int_qs = []
    env_as = []
    ego_as = []
    sur_as = []
    int_as = []
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    model.eval()
    bcpolicy_dir = '/data/full_version/reasoning/model'
    bcpolicy_name = 'aux_attn_s11_0728_120515'
    backbone = torch.load(f"{bcpolicy_dir}/{bcpolicy_name}.pth", weights_only=False)
    backbone.eval()
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning_raw/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            for d in jd.values():
                env_qa = np.array(d['env_qa'])
                ego_qa = np.array(d['ego_qa'])
                sur_qa = np.array(d['sur_qa'])
                int_qa = np.array(d['int_qa'])
                if len(env_qa) > 0:
                    env_qs += env_qa[:, 0].tolist()
                    env_as += env_qa[:, 1].tolist()
                if len(ego_qa) > 0:
                    ego_qs += ego_qa[:, 0].tolist()
                    ego_as += ego_qa[:, 1].tolist()
                if len(sur_qa) > 0:
                    sur_qs += sur_qa[:, 0].tolist()
                    sur_as += sur_qa[:, 1].tolist()
                if len(int_qa) > 0:
                    int_qs += int_qa[:, 0].tolist()
                    int_as += int_qa[:, 1].tolist()
    env_qs = list(set(env_qs))
    env_as = list(set(env_as))
    ego_qs = list(set(ego_qs))
    ego_as = list(set(ego_as))
    sur_qs = list(set(sur_qs))
    sur_as = list(set(sur_as))
    int_qs = list(set(int_qs))
    int_as = list(set(int_as))
    env_q_embeddings  = model.encode(env_qs, convert_to_tensor=True) 
    env_a_embeddings  = model.encode(env_as, convert_to_tensor=True) 
    ego_q_embeddings  = model.encode(ego_qs, convert_to_tensor=True) 
    ego_a_embeddings  = model.encode(ego_as, convert_to_tensor=True) 
    sur_q_embeddings  = model.encode(sur_qs, convert_to_tensor=True) 
    sur_a_embeddings  = model.encode(sur_as, convert_to_tensor=True) 
    int_q_embeddings  = model.encode(int_qs, convert_to_tensor=True) 
    int_a_embeddings  = model.encode(int_as, convert_to_tensor=True) 

    context, *_, = backbone.get_context(obs, all_masks)