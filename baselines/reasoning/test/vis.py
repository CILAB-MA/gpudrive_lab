import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
from gpudrive.integrations.reasoning.dataloader import ReasoningDataset
import os
from torch.utils.data import DataLoader

if __name__ == '__main__':
    qa_names = ['env', 'ego', 'int']
    data_name = f"validation_trajectory_10000.npz"
    data_path = '/data/full_version/processed/final/reasoning'
    with np.load(os.path.join(data_path, data_name), mmap_mode='r') as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
    with np.load(os.path.join(data_path, f"reasoning_question_{data_name}"), mmap_mode='r') as npz:
        questions = np.concatenate([npz[f'{qa_name}_qs'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"reasoning_answer_{data_name}"), mmap_mode='r') as npz:
        pos_answers = np.concatenate([npz[f'{qa_name}_pos_as'] for qa_name in qa_names], axis=1)
        neg_answers = np.concatenate([npz[f'{qa_name}_neg_as'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"nlp_question_{data_name}"), mmap_mode='r') as npz:
        nlp_questions = np.concatenate([npz[f'{qa_name}_qs'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"nlp_answer_{data_name}"), mmap_mode='r') as npz:
        nlp_pos_answers = np.concatenate([npz[f'{qa_name}_pos_as'] for qa_name in qa_names], axis=1)
        nlp_neg_answers = np.concatenate([npz[f'{qa_name}_neg_as'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"reasoning_mask_{data_name}"), mmap_mode='r') as npz:
        qa_masks = np.concatenate([npz[f'{qa_name}_masks'] for qa_name in qa_names], axis=1).astype('bool')
        B, M = questions.shape[:2]
    dataset = ReasoningDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask,
        rollout_len=5, pred_len=1, 
        exp='vanilla', questions=questions, pos=pos_answers, neg=neg_answers,
        qa_masks=qa_masks, questions_nlp=nlp_questions, pos_nlp=nlp_pos_answers,
        neg_nlp=nlp_neg_answers
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True
    )
    del dataset
    model_path = '/data/full_version/reasoning/model/vanilla'
    model_name = 'aux_attn_s11_0728_120515.pth'
    aux_model = torch.load(os.path.join(model_path, model_name))
    aux_model.eval()
    
