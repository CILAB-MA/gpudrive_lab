import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    # DATALOADER
    parser.add_argument('--npz-path', '-npz', type=str, default="/data/full_version/processed/final_rebuttal/training_trajectory_1000_seed0.npz")
    parser.add_argument('--save-path', '-npy', type=str, default="/data/full_version/processed/final_rebuttal/npy")
    args = parser.parse_args()
    
    return args

def preprocess_valid_masks(dead_mask, actions):
    valid_masks = ~dead_mask
    action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
    valid_masks[action_mask] = False
    new_valid_mask = np.zeros((actions.shape[0], 95), dtype=np.bool_)
    new_valid_mask[:, 4:] = valid_masks
    return new_valid_mask

def preprocess_partner_masks(partner_mask):
    partner_mask = np.pad(partner_mask, ((0, 0), (4, 0), (0, 0)), constant_values=2)
    partner_mask = partner_mask == 2
    return partner_mask

def preprocess_road_masks(road_mask):
    road_mask = np.pad(road_mask, ((0, 0), (4, 0), (0, 0)), mode='constant', constant_values=True)
    return road_mask

if __name__ == "__main__":
    args = parse_args()
        
    # make save path
    save_path = os.path.join(args.save_path, os.path.basename(args.npz_path).replace(".npz", ""))
    os.makedirs(save_path, exist_ok=True)
    
    # load npz data
    data = np.load(args.npz_path)
    
    # save as npy
    np.save(save_path + f"/obs.npy", data["obs"])
    np.save(save_path + f"/actions.npy", data["actions"])
    np.save(save_path + f"/valid_mask.npy", preprocess_valid_masks(data["dead_mask"], data["actions"]))
    np.save(save_path + f"/partner_mask.npy", preprocess_partner_masks(data["partner_mask"]))
    np.save(save_path + f"/road_mask.npy", preprocess_road_masks(data["road_mask"]))
    
    

     
    
    
    
    