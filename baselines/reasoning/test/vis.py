import numpy as np
import matplotlib.pyplot as plt
import torch
from gpudrive.integrations.reasoning.dataloader import ReasoningDataset
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd

if __name__ == '__main__':
    qa_names = ['env', 'ego', 'int']
    data_name = f"validation_trajectory_10000.npz"
    data_path = '/data/full_version/processed/final/reasoning'
    exp_name = 'vanilla'
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
    with np.load(os.path.join(data_path, f"nlp_question_{data_name}"), mmap_mode='r', allow_pickle=True) as npz:
        nlp_questions = np.concatenate([npz[f'{qa_name}_qs'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"nlp_answer_{data_name}"), mmap_mode='r', allow_pickle=True) as npz:
        nlp_pos_answers = np.concatenate([npz[f'{qa_name}_pos_as'] for qa_name in qa_names], axis=1)
        nlp_neg_answers = np.concatenate([npz[f'{qa_name}_neg_as'] for qa_name in qa_names], axis=1)
    with np.load(os.path.join(data_path, f"reasoning_mask_{data_name}"), mmap_mode='r') as npz:
        qa_masks = np.concatenate([npz[f'{qa_name}_masks'] for qa_name in qa_names], axis=1).astype('bool')
        B, M = questions.shape[:2]
    dataset = ReasoningDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask,
        rollout_len=5, pred_len=1, exp=exp_name, questions=questions, 
        pos=pos_answers, neg=neg_answers, qa_masks=qa_masks
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
    nlp_pos_answers = nlp_pos_answers.tolist()
    nlp_questions = nlp_questions.tolist()
    losses_all = []
    questions_all = []
    answers_all = []
    for i, batch in enumerate(dataloader):
        batch_size = batch[0].size(0)
        if 'neg' in exp_name:
            obs, expert_action, partner_masks, road_masks, questions, pos, qa_masks, neg, indices, qa_indices = batch
            questions = questions.to("cuda").float()
            pos = pos.to("cuda").float()
            neg = neg.to("cuda").float()
            qa_masks = qa_masks.to("cuda").bool()
        elif exp_name != 'baseline':
            obs, expert_action, partner_masks, road_masks, questions, pos, qa_masks, indices, qa_indices = batch
            questions = questions.to("cuda").float()
            pos = pos.to("cuda").float()
            neg = None
            qa_masks = qa_masks.to("cuda").bool()
        obs, expert_action = obs.to("cuda"), expert_action.to("cuda")
        partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
        road_masks = road_masks.to("cuda") if len(batch) > 3 else None
        all_masks= [partner_masks, road_masks]
        with torch.no_grad():
            context, *_ = aux_model.get_context(obs, all_masks)
            context_repeat = context.unsqueeze(1).repeat(1, questions.shape[1], 1)
            aux_input = torch.cat([context_repeat, questions], dim=-1)
            aux_input = aux_input.reshape(-1, 768)
            qa_masks = qa_masks.reshape(-1)
            pos = pos.reshape(-1, 384)
            pred_answer = aux_model.aux_head(aux_input)

            pred_answer = pred_answer[~qa_masks]
            pos = pos[~qa_masks]
            pos_answer_selected = np.array([
                [nlp_pos_answers[ego_idx][qa_idx] for qa_idx in qa_row]
                for ego_idx, qa_row in zip(indices[:, 0].tolist(), qa_indices.tolist())
            ]).reshape(-1)
            questions_selected = np.array([
                [nlp_questions[ego_idx][qa_idx] for qa_idx in qa_row]
                for ego_idx, qa_row in zip(indices[:, 0].tolist(), qa_indices.tolist())
            ]).reshape(-1)
            qa_masks_np = qa_masks.cpu().numpy().astype('bool')
            pos_answer_selected = pos_answer_selected[~qa_masks_np]
            questions_selected = questions_selected[~qa_masks_np]
            loss = 1 - F.cosine_similarity(pred_answer, pos, dim=-1)
            losses_all.append(loss.cpu())
            questions_all.append(questions_selected)
            answers_all.append(pos_answer_selected)
    # 누적 결과 정리
    losses_all = torch.cat(losses_all, dim=0).numpy()
    questions_all = np.concatenate(questions_all, axis=0)
    answers_all = np.concatenate(answers_all, axis=0)
    # top-50
    top_idx = np.argsort(losses_all)[-50:]
    low_idx = np.argsort(losses_all)[:50]

    top_df = pd.DataFrame({
        'Type': 'High Loss',
        'Loss': losses_all[top_idx],
        'Question': questions_all[top_idx],
        'Answer': answers_all[top_idx],
    })
    low_df = pd.DataFrame({
        'Type': 'Low Loss',
        'Loss': losses_all[low_idx],
        'Question': questions_all[low_idx],
        'Answer': answers_all[low_idx],
    })
    all_df = pd.concat([top_df, low_df])

    from tabulate import tabulate
    print(tabulate(all_df, headers='keys', tablefmt='fancy_grid', showindex=False))