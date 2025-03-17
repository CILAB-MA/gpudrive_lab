import torch
import torch.nn as nn
from algorithms.sb3.dynamic_space.model import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--device', '-d', default='cuda')
    parser.add_argument('--early-stop-num', '-e', default=4)
    args = parser.parse_args()
    
    return args

def train():
    model_path = "/data/RL/model/pred_model"
    env_config = EnvConfig(
        dynamics_model='delta_local',
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-3.14, 3.14, 300), decimals=3
        ),
    )
    # config rl 코드 보고 가져오기
    bc_config = BehavCloningConfig()
    pred_model = LateFusionBCNet()

    with np.load(os.path.join('/data/RL/data/train_trajectory_5000.npz')) as npz:
        train_dataset = ExpertDataset(**npz)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )
    del train_dataset

    with np.load(os.path.join('/data/RL/data/test_trajectory_1000.npz')) as npz:
        test_dataset = ExpertDataset(**npz)
    test_loader = DataLoader(
        test_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )
    del test_dataset
    
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    currenttime = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"action_pred_20step{currenttime}"
    wandb.init(
        project=private_info['main_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id,
        group=f"Pred Model",
        config={**bc_config.__dict__, **env_config.__dict__},
    )
    best_loss = 9999999
    early_stopping = 0
    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        bc_policy.train()
        losses = 0
        mu_losses = 0
        std_losses = 0
        for i, batch in enumerate(train_loader):
            obs, mu, std, _ = batch
            obs, mu, std = obs.to(args.device), mu.to(args.device), std.to(args.device) 
            pred_mu, pred_std = bc_policy(obs) 
            mu_loss = F.smooth_l1_loss(pred_mu, mu)
            std_loss = F.smooth_l1_loss(pred_std, std)
            tot_loss = mu_loss + std_loss
            optimizer.zero_grad()
            tot_loss.mean().backward()
            optimizer.step()
            losses += tot_loss
            mu_losses += mu_loss
            std_losses += std_loss
        if arg.use_wandb:
            log_dict = {
                "train/loss": losses / (i + 1),
                "train/mu_loss": mu_losses / (i + 1),
                "train/std_loss": std_losses / (i + 1),
            }
            wandb.log(log_dict, step=epoch)
        
        if epoch % 5 == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            bc_policy.eval()
            losses = 0
            mu_losses = 0
            std_losses = 0
            for i, batch in enumerate(test_loader):
                with torch.no_grad():
                    obs, mu, std, _ = batch
                    obs, mu, std = obs.to(args.device), mu.to(args.device), std.to(args.device) 
                    pred_mu, pred_std = bc_policy(obs) 

                    mu_loss = F.smooth_l1_loss(pred_mu, mu)
                    std_loss = F.smooth_l1_loss(pred_std, std)
                    losses += tot_loss
                    mu_losses += mu_loss
                    std_losses += std_loss

            if arg.use_wandb:
                log_dict = {
                    "eval/loss": losses / (i + 1),
                    "eval/mu_loss": mu_losses / (i + 1),
                    "eval/std_loss": std_losses / (i + 1),
                }
                wandb.log(log_dict, step=epoch)

            if test_loss < best_loss:
                torch.save(bc_policy, f"{model_path}/pred_model.pth")
                best_loss = losses
                earlt_stopping = 0
                print(f'EPOCH {epoch} gets BEST!')
            else:
                early_stopping += 1
                if early_stopping > args.early_stop_num + 1:
                    wandb.finish()
                    break
if __name__ == '__main__':
    args = parse_args()
    train(args)