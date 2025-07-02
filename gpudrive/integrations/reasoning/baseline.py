import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from torch.optim import AdamW
import wandb
import argparse
from datetime import datetime
from functools import partial
import torch.distributions as dist

def set_seed(seed=42, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

class AuxHead(nn.Module):
    def __init__(self, input_dim, aux_dim=2):
        super(AuxHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.head = nn.Linear(64, aux_dim)
        self.aux_dim = aux_dim
        
    def forward(self, x, deterministic=None):
        x = self.input_layer(x)
        aux_preds = self.head(x)
        return aux_preds

class QADataset(Dataset):
    def __init__(self, question, answer, masks, obs=None, obs_masks=None, exp_model='baseline',
                 start_idx=0):
        B, MAX_LEN = question.shape[:2]
        self.qs = torch.from_numpy(question[~masks]).float()
        self.as_ = torch.from_numpy(answer[~masks]).float()
        if obs is not None:
            self._obs = torch.from_numpy(obs[:, start_idx:start_idx + 5]).float()
            self._obs = self._obs.reshape(B, -1).unsqueeze(1)
            self._obs = self._obs.repeat(1, MAX_LEN, 1)
            self._obs = self._obs[~masks]
            partner_mask = obs_masks[0]
            road_mask = obs_masks[1]
            self._partner_mask = torch.from_numpy(partner_mask[:, start_idx + 5 -1]).float()
            self._partner_mask = self._partner_mask.reshape(B, -1).unsqueeze(1)
            self._partner_mask = self._partner_mask.repeat(1, MAX_LEN, 1)
            self._partner_mask = self._partner_mask[~masks]
            self._partner_mask = self._partner_mask == 2
            self._road_mask = torch.from_numpy(road_mask[:, 4])
            self._road_mask = self._road_mask.reshape(B, -1).unsqueeze(1)
            self._road_mask = self._road_mask.repeat(1, MAX_LEN, 1)
            self._road_mask = self._road_mask[~masks]
        self.exp_model = exp_model
        del obs

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        if self.exp_model == 'baseline':
            return self.qs[idx], self.as_[idx]
        else:
            return self.qs[idx], self.as_[idx], self._obs[idx], self._partner_mask[idx], self._road_mask[idx]
        
def get_dataloader(data_path, data_file, exp_name='env', isshuffle=True, traj_file=None, 
                   model='baseline', start_idx=0):
    with np.load(os.path.join(data_path, data_file), mmap_mode='r') as npz:
        questions = npz[f'{exp_name}_qs']
        answers = npz[f'{exp_name}_as']
        masks = npz[f'{exp_name}_masks']
        B, M = questions.shape[:2]
        concat_vecs = np.concatenate([questions, answers], axis=-1)
        flat_vecs = concat_vecs.reshape(-1, 768)
        _, unique_indices = np.unique(flat_vecs, axis=0, return_index=True)
        unique_mask_flat = np.zeros(flat_vecs.shape[0], dtype=bool)
        unique_mask_flat[unique_indices] = True
        unique_mask = unique_mask_flat.reshape(B, M)
        final_mask = ~((unique_mask == True) & (masks == False))

    obs = None
    obs_masks = None
    if model != 'baseline':
         with np.load(os.path.join(data_path, traj_file), mmap_mode='r') as npz:
             obs = npz['obs']
             partner_mask = npz['partner_mask']
             road_mask = npz['road_mask']
             obs_masks = [partner_mask, road_mask]

    dataset = QADataset(questions, answers, final_mask, obs, obs_masks=obs_masks, exp_model=model,
                        start_idx=start_idx)
    data_len = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=isshuffle,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True
    )
    del dataset
    return dataloader, data_len

def evaluate(dataloader, model, bc_policy, exp_model='baseline'):
    eval_losses = 0
    for i, batch in enumerate(dataloader):
        if exp_model == 'baseline':
            question, answer = batch
        else:
            question, answer, obs, partner_masks, road_masks = batch
            obs = obs.cuda()
            partner_masks = partner_masks.cuda().unsqueeze(1)
            road_masks = road_masks.cuda().unsqueeze(1)
            all_masks= [partner_masks, road_masks]
            context, *_  = bc_policy.get_context(obs, all_masks)
        question = question.cuda()
        answer = answer.cuda()
        if exp_model != 'baseline':
            question = torch.cat([question, context], dim=-1)
        with torch.no_grad():
            pred_answer = model(question)
            loss = 1 - F.cosine_similarity(pred_answer, answer, dim=-1).mean()
        eval_losses += loss
    return eval_losses / (i + 1)

def train(args):
    data_path = '/data/full_version/processed/final/reasoning'
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    traj_train_file = None
    traj_valid_file = None
    if args.use_wandb:
        wandb.init()
        exp_name = dict(wandb.config)['qa_name']
        exp_model = dict(wandb.config)['model_name']
        start_idx = dict(wandb.config)['start_idx']
        seed = dict(wandb.config)['seed']
        wandb.run.name = f'base_{exp_name}_{current_time}'
        wandb.run.save()
        
    else:
        exp_name = 'env'
        exp_model = 'baseline'
        seed = 42
        start_idx = 0
    if exp_model != 'baseline':
        traj_train_file = "training_trajectory_80000.npz"
        traj_valid_file = "validation_trajectory_10000.npz"
    tr_loader, tr_len = get_dataloader(data_path, "reasoning_training_trajectory_80000.npz", exp_name=exp_name,
                                       traj_file=traj_train_file, model=exp_model, start_idx=start_idx)
    te_loader, te_len = get_dataloader(data_path, "reasoning_validation_trajectory_10000.npz", exp_name=exp_name, 
                               isshuffle=False, traj_file=traj_valid_file, model=exp_model, start_idx=start_idx)
    bc_policy = None
    if exp_model == 'pretrained':
        model_path = '/data/full_version/model/cov1792_clip10/early_attn_s3_0630_072820_60000.pth'
        bc_policy = torch.load(model_path, weights_only=False).to("cuda")
        bc_policy.eval()
    gradient_steps = 0
    set_seed(seed)
    input_dim = 384 if exp_model == 'baseline' else 768
    model = AuxHead(input_dim, 384)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    model = model.cuda()
    if args.use_wandb:
        wandb.run.tags = tuple([f'num_train_{tr_len}', f'num_test_{te_len}', f'param_{trainable_params}'])
        wandb.run.save()
    pbar = tqdm(total=20000, desc="Gradient Steps", ncols=100)
    optimizer = AdamW(model.parameters(), lr=4e-4, eps=0.0001)
    while gradient_steps < 20000:
        train_losses = 0
        for n, batch in enumerate(tr_loader):
            if exp_model == 'baseline':
                question, answer = batch
            else:
                question, answer, obs, partner_masks, road_masks = batch
                obs = obs.cuda()
                partner_masks = partner_masks.cuda().unsqueeze(1)
                road_masks = road_masks.cuda().unsqueeze(1)
                all_masks= [partner_masks, road_masks]
                context, *_  = bc_policy.get_context(obs, all_masks)
            question = question.cuda()
            answer = answer.cuda()
            if exp_model != 'baseline':
                question = torch.cat([question, context], dim=-1)
            pred_answer = model(question)
            loss = 1 - F.cosine_similarity(pred_answer, answer, dim=-1).mean()
            # loss = 1 - F.cosine_similarity(pred_answer, answer, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            gradient_steps += 1
            pbar.update(1)
            train_losses += loss.item()

            if gradient_steps % 500 == 0:
                model.eval()
                eval_loss = evaluate(te_loader, model, bc_policy, exp_model)
                if args.use_wandb:
                    wandb.log({'eval/loss': eval_loss}, step=gradient_steps)

        train_loss = train_losses / (n + 1)
        if args.use_wandb:
            wandb.log({'train/loss': train_loss}, step=gradient_steps)
    wandb.finish()
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Most of vars are in il.yaml. These are for different server.")
    # EXPERIMENT
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--sweep-id', type=str, default=None)
    args = parser.parse_args()
    if args.use_wandb:
        import yaml
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        with open("gpudrive/integrations/reasoning/sweep.yaml") as f:
            exp_config = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        train_fn = partial(train, args=args)
        if args.sweep_id is not None:
            wandb.agent(args.sweep_id, function=train_fn, project=private_info['reasoning_project'], entity=private_info['entity'])
        else:
            sweep_id = wandb.sweep(exp_config, project=private_info['reasoning_project'], entity=private_info['entity'])
            wandb.agent(sweep_id, function=train_fn)
    else:
        train(args)