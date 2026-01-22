import argparse
import numpy as np
import os
import shutil
import gc  # 가비지 컬렉션용

ROLL_OUT_LEN = 5
PRED_LEN = 1

# --------------------------------------------------
# Argument
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-path', type=str,
                        default="/data/full_version/final/training_trajectory_80000.npz")
    parser.add_argument('--save-path', type=str,
                        default="/data/full_version/final_rebuttal/npy")
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

    print(f"Loading npz from {args.npz_path}...")
    # 600GB 로드 (RAM 60% 사용)
    data = np.load(args.npz_path)
    print("NPZ loaded.")

    # 필요한 데이터만 참조 후 data 객체 삭제 유도 가능하면 좋음
    obs = data["obs"]
    actions = data["actions"]
    
    # 마스크 전처리
    print("Preprocessing masks...")
    valid_mask = preprocess_valid_masks(data["dead_mask"], actions)
    partner_mask = preprocess_partner_masks(data["partner_mask"])
    road_mask = preprocess_road_masks(data["road_mask"])

    # data 객체 명시적 삭제로 메모리 확보 시도 (obs, actions 등은 참조 유지)
    del data
    gc.collect()

    B, T, F = obs.shape
    A = actions.shape[-1]

    print("Counting valid samples...")
    count = 0
    # 유효 샘플 개수 카운팅 (수정 없음)
    for b in range(B):
        for t in range(T - (ROLL_OUT_LEN + PRED_LEN - 2)):
            if valid_mask[b, t + ROLL_OUT_LEN + PRED_LEN - 2]:
                count += 1

    print(f"Total valid samples: {count}")
    
    # 예상 용량 계산 및 경고
    estimated_size_gb = (count * ROLL_OUT_LEN * F * 4) / (1024**3)  # float32 = 4 bytes
    print(f"Estimated Output Size: {estimated_size_gb:.2f} GB")
    print("⚠️ Ensure you have enough DISK space (not RAM).")

    # --------------------------------------------------
    # ⭐ 핵심 수정: np.lib.format.open_memmap 사용
    # --------------------------------------------------
    # 이 함수는 .npy 헤더를 포함하여 파일을 생성하므로, 
    # 나중에 변환할 필요 없이 작업이 끝나면 바로 유효한 .npy 파일이 됩니다.
    
    print("Creating npy files directly on disk...")
    
    obs_mm = np.lib.format.open_memmap(
        os.path.join(save_dir, "obs.npy"),
        mode='w+', dtype=np.float32, shape=(count, ROLL_OUT_LEN, F)
    )

    actions_mm = np.lib.format.open_memmap(
        os.path.join(save_dir, "actions.npy"),
        mode='w+', dtype=np.float32, shape=(count, PRED_LEN, A)
    )

    partner_mm = np.lib.format.open_memmap(
        os.path.join(save_dir, "partner_mask.npy"),
        mode='w+', dtype=np.bool_, shape=(count, ROLL_OUT_LEN, partner_mask.shape[-1])
    )

    road_mm = np.lib.format.open_memmap(
        os.path.join(save_dir, "road_mask.npy"),
        mode='w+', dtype=np.bool_, shape=(count, ROLL_OUT_LEN, road_mask.shape[-1])
    )

    print("Writing data to npy files...")
    idx = 0
    for b in range(B):
        # 진행 상황 로깅 (대용량 처리시 필수)
        if b % 100 == 0:
            print(f"Processing batch {b}/{B}...")
            
        for t in range(T - (ROLL_OUT_LEN + PRED_LEN - 2)):
            if not valid_mask[b, t + ROLL_OUT_LEN + PRED_LEN - 2]:
                continue

            # RAM(obs) -> Disk(obs_mm) 복사
            # OS가 알아서 페이지 캐싱을 관리하므로 RAM이 터지지 않음
            obs_mm[idx] = obs[b, t:t + ROLL_OUT_LEN]
            actions_mm[idx] = actions[b, t:t + PRED_LEN]
            partner_mm[idx] = partner_mask[b, t:t + ROLL_OUT_LEN]
            road_mm[idx] = road_mask[b, t:t + ROLL_OUT_LEN]

            idx += 1

    # 변경사항 디스크 동기화 (선택사항, 안전을 위해)
    obs_mm.flush()
    actions_mm.flush()
    partner_mm.flush()
    road_mm.flush()

    print("Saved final npy files successfully.")
    print("Preprocessing finished.")