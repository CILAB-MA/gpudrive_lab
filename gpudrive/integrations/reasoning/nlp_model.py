import os
import sys
import json
import argparse
import numpy as np

from tqdm import tqdm
import torch
import numpy as np
import os
import re
from collections import defaultdict
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=80000)
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
    int_qas = []
    sur_qas = []
    env_qas = []
    ego_qas = []
    qa_types = ["env", "ego", "sur", "int"]
    save_path = f'/data/full_version/reasoning/processed/{args.data_dir}_subset'
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    input_text = "The weather is nice today."
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with np.load(f"{save_path}/reasoning/reasoning_trajectory_{idx * args.scene_batch_size}.npz", mmap_mode='r') as npz:
                env_mask = npz['env_mask']
                ego_mask = npz['ego_mask']
                sur_mask = npz['sur_mask']
                int_mask = npz['int_mask']
            with np.load(f"{save_path}/nlp/reasoning_trajectory_{idx * args.scene_batch_size}.npz", mmap_mode='r', allow_pickle=True) as nlp:
                env_q_nlp = nlp['env_q']
                env_pos_nlp = nlp['env_pos_a']
                env_neg_nlp = nlp['env_neg_a']
                ego_q_nlp = nlp['ego_q']
                ego_pos_nlp = nlp['ego_pos_a']
                ego_neg_nlp = nlp['ego_neg_a']
                sur_q_nlp = nlp['sur_q']
                sur_pos_nlp = nlp['sur_pos_a']
                sur_neg_nlp = nlp['sur_neg_a']
                int_q_nlp = nlp['int_q']
                int_pos_nlp = nlp['int_pos_a']
                int_neg_nlp = nlp['int_neg_a']
            valid_agent = len(env_mask)
            expert_env_q_lst = np.zeros((valid_agent, 20, 384))
            expert_ego_q_lst = np.zeros((valid_agent, 20, 384))
            expert_sur_q_lst = np.zeros((valid_agent, 120, 384))
            expert_int_q_lst = np.zeros((valid_agent, 30, 384))
            expert_env_ap_lst = np.zeros((valid_agent, 20, 384))
            expert_ego_ap_lst = np.zeros((valid_agent, 20, 384))
            expert_sur_ap_lst = np.zeros((valid_agent, 120, 384))
            expert_int_ap_lst = np.zeros((valid_agent, 30, 384))
            expert_env_an_lst = np.zeros((valid_agent, 20, 384))
            expert_ego_an_lst = np.zeros((valid_agent, 20, 384))
            expert_sur_an_lst = np.zeros((valid_agent, 120, 384))
            expert_int_an_lst = np.zeros((valid_agent, 30, 384))
            q_nlp = [env_q_nlp, ego_q_nlp, sur_q_nlp, int_q_nlp]
            ap_nlp = [env_pos_nlp, ego_pos_nlp, sur_pos_nlp, int_pos_nlp]
            an_nlp = [env_neg_nlp, ego_neg_nlp, sur_neg_nlp, int_neg_nlp]
            for q, ap, an in zip(q_nlp, ap_nlp, an_nlp):
                q_inputs = tokenizer(q, return_tensors="pt")
                ap_inputs = tokenizer(ap, return_tensors="pt")
                an_inputs = tokenizer(an, return_tensors="pt")
                # Encoder hidden states as sentence embedding
                with torch.no_grad():
                    q_outputs = model.encoder(**q_inputs)
                    q_embedding = q_outputs.last_hidden_state.mean(dim=1)  # mean pooling
                    ap_outputs = model.encoder(**ap_inputs)
                    ap_embedding = ap_outputs.last_hidden_state.mean(dim=1)  # mean pooling
                    an_outputs = model.encoder(**an_inputs)
                    an_embedding = an_outputs.last_hidden_state.mean(dim=1)  # mean pooling
                    
            q_npy = [expert_env_q_lst, expert_ego_q_lst, expert_sur_q_lst, expert_int_q_lst]
            ap_npy = [expert_env_ap_lst, expert_ego_ap_lst, expert_sur_ap_lst, expert_int_ap_lst]
            an_npy = [expert_env_an_lst, expert_ego_an_lst, expert_sur_an_lst, expert_int_an_lst]

            expert_env_q_lst = q_npy[0]
            expert_ego_q_lst = q_npy[1]
            expert_sur_q_lst = q_npy[2]
            expert_int_q_lst = q_npy[3]

            expert_env_ap_lst = ap_npy[0]
            expert_ego_ap_lst = ap_npy[1]
            expert_sur_ap_lst = ap_npy[2]
            expert_int_ap_lst = ap_npy[3]

            expert_env_an_lst = an_npy[0]
            expert_ego_an_lst = an_npy[1]
            expert_sur_an_lst = an_npy[2]
            expert_int_an_lst = an_npy[3]

            save_path = f'/data/full_version/processed/final/reasoning_{args.data_dir}_subset'
            os.makedirs(save_path + '/reasoning_posneg/nlp', exist_ok=True)
            np.savez_compressed(f"{save_path}/reasoning_posneg/nlp/reasoning_trajectory_{idx * args.scene_batch_size}.npz", 
                    env_q=expert_env_q_lst,
                    ego_q=expert_ego_q_lst,
                    sur_q=expert_sur_q_lst,
                    int_q=expert_int_q_lst,
                    env_pos_a=expert_env_ap_lst,
                    ego_pos_a=expert_ego_ap_lst,
                    sur_pos_a=expert_sur_ap_lst,
                    int_pos_a=expert_int_ap_lst,
                    env_neg_a=expert_env_an_lst,
                    ego_neg_a=expert_ego_an_lst,
                    sur_neg_a=expert_sur_an_lst,
                    int_neg_a=expert_int_an_lst,
                    )