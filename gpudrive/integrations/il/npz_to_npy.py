import argparse
import numpy as np
import os
import gc
import sys
from tqdm import tqdm  # tqdm 임포트

ROLL_OUT_LEN = 5
PRED_LEN = 1

# --------------------------------------------------
# Argument
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default="/data/full_version/training_subset_v5") 
    parser.add_argument('--save-path', type=str,
                        default="/data/full_version/final_rebuttal/npy/training_trajectory_80000")
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

# --------------------------------------------------
# File List Generator
# --------------------------------------------------
def get_file_list(data_dir):
    file_list = []
    # 0부터 79900까지 100단위
    for i in range(0, 80000, 100):
        filename = f"trajectory_{i}.npz"
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            file_list.append(path)
        # 존재하지 않는 파일 경고는 너무 많을 수 있으므로 생략하거나 필요한 경우 주석 해제
        # else:
        #    print(f"Warning: {filename} not found.")
    return file_list

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    
    npz_files = get_file_list(args.data_dir)
    print(f"Found {len(npz_files)} npz files to process.")

    if len(npz_files) == 0:
        print("No files found. Exiting.")
        sys.exit()

    # ==================================================================
    # [Pass 1] 총 유효 샘플 수 계산
    # ==================================================================
    print("\n>>> [Pass 1/2] Counting total valid samples...")
    
    total_count = 0
    
    # Shape 확인용 임시 로드
    temp_data = np.load(npz_files[0])
    obs_shape_F = temp_data["obs"].shape[-1]
    act_shape_A = temp_data["actions"].shape[-1]
    partner_last_dim = preprocess_partner_masks(temp_data["partner_mask"]).shape[-1]
    road_last_dim = preprocess_road_masks(temp_data["road_mask"]).shape[-1]
    del temp_data
    gc.collect()

    # TQDM 적용: Pass 1
    # desc: 진행바 제목, unit: 단위 표시
    pbar1 = tqdm(npz_files, desc="Counting", unit="file")
    
    for fpath in pbar1:
        try:
            with np.load(fpath) as data:
                actions = data["actions"]
                dead_mask = data["dead_mask"]
                
                valid_mask = preprocess_valid_masks(dead_mask, actions)
                
                B, T, _ = actions.shape
                
                for b in range(B):
                    time_indices = np.arange(T - (ROLL_OUT_LEN + PRED_LEN - 2))
                    target_indices = time_indices + ROLL_OUT_LEN + PRED_LEN - 2
                    
                    valid_count_b = np.sum(valid_mask[b, target_indices])
                    total_count += valid_count_b
            
            # 진행바 옆에 현재 누적 카운트 표시 (실시간 확인 가능)
            pbar1.set_postfix(samples=total_count)

        except Exception as e:
            # tqdm 사용 중 print를 쓰면 UI가 깨지므로 pbar.write 사용
            pbar1.write(f"Error reading {fpath}: {e}")
            continue

    print(f"\nTotal valid samples calculated: {total_count}")
    estimated_size_gb = (total_count * ROLL_OUT_LEN * obs_shape_F * 4) / (1024**3)
    print(f"Estimated Obs.npy Size: {estimated_size_gb:.2f} GB")

    # ==================================================================
    # [Memory Mapping] 파일 생성
    # ==================================================================
    print("\n>>> Creating memmap files...")
    
    obs_mm = np.lib.format.open_memmap(
        os.path.join(args.save_path, "obs.npy"),
        mode='w+', dtype='float32', shape=(int(total_count), ROLL_OUT_LEN, obs_shape_F)
    )

    actions_mm = np.lib.format.open_memmap(
        os.path.join(args.save_path, "actions.npy"),
        mode='w+', dtype='float32', shape=(int(total_count), PRED_LEN, act_shape_A)
    )

    partner_mm = np.lib.format.open_memmap(
        os.path.join(args.save_path, "partner_mask.npy"),
        mode='w+', dtype='bool', shape=(int(total_count), ROLL_OUT_LEN, partner_last_dim)
    )

    road_mm = np.lib.format.open_memmap(
        os.path.join(args.save_path, "road_mask.npy"),
        mode='w+', dtype='bool', shape=(int(total_count), ROLL_OUT_LEN, road_last_dim)
    )

    # ==================================================================
    # [Pass 2] 데이터 쓰기
    # ==================================================================
    print("\n>>> [Pass 2/2] Writing data to disk...")
    
    global_idx = 0
    
    # TQDM 적용: Pass 2
    pbar2 = tqdm(npz_files, desc="Writing", unit="file")

    for fpath in pbar2:
        try:
            data = np.load(fpath)
            
            obs = data["obs"]
            actions = data["actions"]
            
            valid_mask = preprocess_valid_masks(data["dead_mask"], actions)
            partner_mask = preprocess_partner_masks(data["partner_mask"])
            road_mask = preprocess_road_masks(data["road_mask"])
            
            del data

            B, T, _ = obs.shape

            for b in range(B):
                time_steps = T - (ROLL_OUT_LEN + PRED_LEN - 2)
                check_indices = np.arange(time_steps) + ROLL_OUT_LEN + PRED_LEN - 2
                is_valid = valid_mask[b, check_indices]
                
                num_valid = np.sum(is_valid)
                if num_valid == 0:
                    continue

                valid_t_indices = np.where(is_valid)[0]

                for t in valid_t_indices:
                    obs_mm[global_idx] = obs[b, t:t + ROLL_OUT_LEN]
                    actions_mm[global_idx] = actions[b, t:t + PRED_LEN]
                    partner_mm[global_idx] = partner_mask[b, t:t + ROLL_OUT_LEN]
                    road_mm[global_idx] = road_mask[b, t:t + ROLL_OUT_LEN]
                    
                    global_idx += 1
            
            del obs, actions, valid_mask, partner_mask, road_mask
            gc.collect()

            # 진행바 옆에 현재 저장된 인덱스 표시
            pbar2.set_postfix(saved=global_idx)

        except Exception as e:
            pbar2.write(f"Error processing {fpath}: {e}")
            continue

    # Flush
    obs_mm.flush()
    actions_mm.flush()
    partner_mm.flush()
    road_mm.flush()

    pbar2.close() # tqdm 종료 (for문 끝나면 자동이지만 명시적으로)

    print("\nAll done!")
    print(f"Final Global Index: {global_idx}")
    print(f"Expected Count: {total_count}")
    
    assert global_idx == total_count, "Index Mismatch Error! Something went wrong."