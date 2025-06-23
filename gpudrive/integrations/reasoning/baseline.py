import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
from torch.optim import AdamW
import wandb
import argparse
from datetime import datetime
from functools import partial

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
            nn.Linear(input_dim, 384),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.head = nn.Linear(384, aux_dim)
        self.aux_dim = aux_dim
        
    def forward(self, x, deterministic=None):
        x = self.input_layer(x)
        aux_preds = self.head(x)
        return aux_preds

class QADataset(Dataset):
    def __init__(self, question, answer):

        self.qs = torch.from_numpy(question).float()
        self.as_ = torch.from_numpy(answer).float()

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, idx):
        return self.qs[idx], self.as_[idx]

def get_dataloader(data_path, data_file, exp_name='env', isshuffle=True):
    with np.load(os.path.join(data_path, data_file), mmap_mode='r') as npz:
        questions = npz[f'{exp_name}_q']
        answers = npz[f'{exp_name}_a']

    dataset = QADataset(questions, answers)
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

def evaluate(dataloader, model):
    eval_losses = 0
    for i, batch in enumerate(dataloader):
        question, answer = batch
        question = question.cuda()
        answer = answer.cuda()
        with torch.no_grad():
            pred_answer = model(question)
            loss = 1 - F.cosine_similarity(pred_answer, answer, dim=-1).mean()
        eval_losses += loss
    return eval_losses / (i + 1)

def train(args):
    data_path = '/data/full_version/processed/final'
    current_time = datetime.now().strftime("%m%d_%H%M%S")

    if args.use_wandb:
        wandb.init()
        exp_name = dict(wandb.config)['qa_name']
        seed = dict(wandb.config)['seed']
        wandb.run.name = f'base_{exp_name}_{current_time}'
        wandb.run.save()
    else:
        exp_name = 'env'
    tr_loader, tr_len = get_dataloader(data_path, "womd_reasoning_embed_training.npz", exp_name=exp_name)
    te_loader, te_len = get_dataloader(data_path, "womd_reasoning_embed_validation.npz", exp_name=exp_name, 
                               isshuffle=False)
    if args.use_wandb:
        wandb.run.tags = tuple([f'num_train_{tr_len}', f'num_test_{te_len}'])
        wandb.run.save()
    gradient_steps = 0
    set_seed(seed)
    model = AuxHead(384, 384)
    model = model.cuda()
    pbar = tqdm(total=20000, desc="Gradient Steps", ncols=100)
    optimizer = AdamW(model.parameters(), lr=3e-4, eps=0.0001)
    while gradient_steps < 20000:
        train_losses = 0
        for n, batch in enumerate(tr_loader):
            question, answer = batch
            question = question.cuda()
            answer = answer.cuda()
            pred_answer = model(question)

            loss = 1 - F.cosine_similarity(pred_answer, answer, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            gradient_steps += 1
            pbar.update(1)
            train_losses += loss.item()

            if gradient_steps % 500 == 0:
                model.eval()
                eval_loss = evaluate(te_loader, model)
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