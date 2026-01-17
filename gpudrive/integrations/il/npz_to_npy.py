import argparse
import numpy as np
import os
import shutil

ROLL_OUT_LEN = 5
PRED_LEN = 1
CHUNK_SIZE = 10000  # memmap → npy 변환 시 chunk 크기

# --------------------------------------------------
# Argument
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-path', type=str,
                        default="/data/full_version/processed/final_rebuttal/training_trajectory_500_seed0.npz")
    parser.add_argument('--save-path', type=str,
                        default="/data/full_version/processed/final_rebuttal/npy")
    return parser.parse_args()

# --------------------------------------------------
# Preprocess functions
# --------------------------------------------------
def preprocess_valid_masks(dead_mask, actions):
    valid_masks = ~dead_mask
    action_mask = (
        (np.abs(actions[..., 1]) > 0.5) |
        (np.abs(actions[..., 0]) > 5) |
        (np.abs(actions[..., -1]) > 0.2)
    )
    valid_masks[action_mask] = False

    new_valid_mask = np.zeros((actions.shape[0], 95), dtype=np.bool_)
    new_valid_mask[:, 4:] = valid_masks
    return new_valid_mask

def preprocess_partner_masks(partner_mask):
    partner_mask = np.pad(partner_mask, ((0, 0), (4, 0), (0, 0)),
                           constant_values=2)
    return partner_mask == 2

def preprocess_road_masks(road_mask):
    return np.pad(road_mask, ((0, 0), (4, 0), (0, 0)),
                  mode='constant', constant_values=True)


if __name__ == "__main__":
    args = parse_args()

    save_dir = os.path.join(
        args.save_path,
        os.path.basename(args.npz_path).replace(".npz", "")
    )
    os.makedirs(save_dir, exist_ok=True)

    print("Loading npz...")
    data = np.load(args.npz_path)

    obs = data["obs"]
    actions = data["actions"]

    valid_mask = preprocess_valid_masks(data["dead_mask"], actions)
    partner_mask = preprocess_partner_masks(data["partner_mask"])
    road_mask = preprocess_road_masks(data["road_mask"])

    B, T, F = obs.shape
    A = actions.shape[-1]

    print("Counting valid samples...")
    count = 0
    for b in range(B):
        for t in range(T - (ROLL_OUT_LEN + PRED_LEN - 2)):
            if valid_mask[b, t + ROLL_OUT_LEN + PRED_LEN - 2]:
                count += 1

    print(f"Total valid samples: {count}")

    tmp_dir = os.path.join(save_dir, "tmp_memmap")
    os.makedirs(tmp_dir, exist_ok=True)

    obs_mm = np.memmap(
        os.path.join(tmp_dir, "obs.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(count, ROLL_OUT_LEN, F)
    )

    actions_mm = np.memmap(
        os.path.join(tmp_dir, "actions.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(count, PRED_LEN, A)
    )

    partner_mm = np.memmap(
        os.path.join(tmp_dir, "partner_mask.dat"),
        dtype=np.bool_,
        mode="w+",
        shape=(count, ROLL_OUT_LEN, partner_mask.shape[-1])
    )

    road_mm = np.memmap(
        os.path.join(tmp_dir, "road_mask.dat"),
        dtype=np.bool_,
        mode="w+",
        shape=(count, ROLL_OUT_LEN, road_mask.shape[-1])
    )

    print("Writing memmap...")
    idx = 0
    for b in range(B):
        for t in range(T - (ROLL_OUT_LEN + PRED_LEN - 2)):
            if not valid_mask[b, t + ROLL_OUT_LEN + PRED_LEN - 2]:
                continue

            obs_mm[idx] = obs[b, t:t + ROLL_OUT_LEN]
            actions_mm[idx] = actions[b, t:t + PRED_LEN]
            partner_mm[idx] = partner_mask[b, t:t + ROLL_OUT_LEN]
            road_mm[idx] = road_mask[b, t:t + ROLL_OUT_LEN]

            idx += 1

    obs_mm.flush()
    actions_mm.flush()
    partner_mm.flush()
    road_mm.flush()

    print("Converting memmap → npy...")

    def memmap_to_npy(mm, out_path):
        arr = np.empty(mm.shape, dtype=mm.dtype)
        for i in range(0, mm.shape[0], CHUNK_SIZE):
            arr[i:i + CHUNK_SIZE] = mm[i:i + CHUNK_SIZE]
        np.save(out_path, arr)

    memmap_to_npy(obs_mm, os.path.join(save_dir, "obs.npy"))
    memmap_to_npy(actions_mm, os.path.join(save_dir, "actions.npy"))
    memmap_to_npy(partner_mm, os.path.join(save_dir, "partner_mask.npy"))
    memmap_to_npy(road_mm, os.path.join(save_dir, "road_mask.npy"))

    print("Saved final npy files.")

    
    # ⭐ memmap 파일 핸들 해제
    del obs_mm
    del actions_mm
    del partner_mm
    del road_mm
    import gc
    gc.collect()

    shutil.rmtree(tmp_dir)
    print("Temporary memmap files removed.")

    print("Preprocessing finished successfully.")