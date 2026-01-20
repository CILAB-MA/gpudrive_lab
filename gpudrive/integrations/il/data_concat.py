import os
import psutil
import numpy as np
from tqdm import tqdm
import gc
import random
import tempfile
import argparse

def print_memory_usage():
    """현재 RAM 사용량을 출력합니다."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    used_gb = mem.used / (1024**3)
    print(f"[RAM] Total: {total_gb:.2f} GB | Used: {used_gb:.2f} GB")

def extract_scene_id(filename):
    return int(filename.split("_")[1].split(".")[0])

def compute_scene_unit(scene_ids):
    diffs = [scene_ids[i+1] - scene_ids[i] for i in range(len(scene_ids)-1)]
    return min(diffs)

def save_large_npz(file_paths, keys_to_process, save_path, save_name):
    """
    대용량 npz 파일들을 메모리 매핑을 사용하여 병합하고 압축 저장합니다.
    """
    if not file_paths:
        print(f"[WARN] 처리할 파일이 없습니다: {save_name}")
        return

    full_save_path = os.path.join(save_path, save_name)
    print(f"\n[Processing] {save_name} (Files: {len(file_paths)})")
    
    # --- Pass 1: 전체 크기(Shape) 계산 ---
    print("  >> Pass 1: Scanning shapes...")
    shapes = {}
    dtypes = {}
    total_rows = {key: 0 for key in keys_to_process}

    # 첫 번째 파일로 shape/dtype 초기화 및 전체 행 수 계산
    for file_path in tqdm(file_paths, desc="Scanning"):
        try:
            # mmap_mode='r'을 사용하여 헤더만 읽거나 필요한 부분만 로드
            with np.load(file_path, mmap_mode="r") as data:
                for key in keys_to_process:
                    if key not in data:
                        continue
                    
                    if key not in shapes:
                        shapes[key] = data[key].shape[1:] # 첫 차원(배치)을 제외한 나머지 차원
                        dtypes[key] = data[key].dtype
                    
                    total_rows[key] += data[key].shape[0]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"  >> Total rows detected: {total_rows}")
    print_memory_usage()

    # --- Pass 2: 임시 파일(Memmap) 생성 및 데이터 채우기 ---
    # tempfile을 사용하여 작업이 끝나거나 에러 발생 시 임시 파일 자동 삭제
    with tempfile.TemporaryDirectory() as tmpdir:
        memmaps = {}
        # Memmap 생성
        for key in keys_to_process:
            if total_rows[key] == 0:
                continue
            final_shape = (total_rows[key],) + shapes[key]
            mmap_path = os.path.join(tmpdir, f"{key}.dat")
            memmaps[key] = np.memmap(mmap_path, dtype=dtypes[key], mode='w+', shape=final_shape)

        cursors = {key: 0 for key in keys_to_process}

        print("  >> Pass 2: Aggregating data into memmap...")
        for file_path in tqdm(file_paths, desc="Aggregating"):
            try:
                with np.load(file_path, mmap_mode="r") as data:
                    for key in keys_to_process:
                        if key not in memmaps: 
                            continue
                        
                        chunk = data[key]
                        count = chunk.shape[0]
                        start = cursors[key]
                        end = start + count
                        
                        # Memmap에 데이터 복사
                        memmaps[key][start:end] = chunk
                        cursors[key] = end
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # --- Pass 3: 압축 저장 (Compress & Save) ---
        print("  >> Pass 3: Saving compressed file...")
        print_memory_usage()
        
        # 실제 데이터가 들어있는 memmap 객체들만 모아서 저장
        save_dict = {key: memmaps[key] for key in keys_to_process if key in memmaps}
        np.savez_compressed(full_save_path, **save_dict)
        
        # 명시적으로 memmap 닫기 (Windows 등에서 파일 잠금 방지)
        for key in memmaps:
            del memmaps[key]
        gc.collect()

    print(f"[DONE] Saved to {full_save_path}")

def run(save_path, save_name, subset_path, num_scenes, seed=42):
    # 1. 파일 리스트 확보 및 Scene ID 추출
    all_files = [f for f in os.listdir(subset_path) if f.endswith(".npz") and f.startswith("trajectory_")]
    
    if not all_files:
        raise ValueError(f"No trajectory files found in {subset_path}")
        
    scene_ids = sorted([extract_scene_id(f) for f in all_files])

    if len(scene_ids) < 2:
        raise ValueError("Num of scene is less than 2.")

    scene_unit = compute_scene_unit(scene_ids)
    print(f"[INFO] scene_unit = {scene_unit}")

    # 2. 필요한 파일 개수 계산 및 랜덤 샘플링
    need_files = (num_scenes + scene_unit - 1) // scene_unit

    if need_files > len(scene_ids):
        print(f"[WARN] Requested {need_files} files, but only {len(scene_ids)} available. Using all.")
        need_files = len(scene_ids)

    random.seed(seed)
    chosen_ids = random.sample(scene_ids, need_files)
    chosen_ids = sorted(chosen_ids)

    print(f"[INFO] Selected {len(chosen_ids)} scenes (Seed: {seed})")

    # 3. 파일 경로 리스트 생성 (메인 데이터 & 글로벌 데이터)
    trajectory_files = []
    global_files = []

    for sid in chosen_ids:
        # Main trajectory file path
        traj_name = f"trajectory_{sid}.npz"
        trajectory_files.append(os.path.join(subset_path, traj_name))

        # Global file path
        global_name = f"global_trajectory_{sid}.npz" # 원본 코드 로직 참조: global_ + filename
        global_files.append(os.path.join(subset_path, "global", global_name))

    # 4. 저장 경로 생성
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "global"), exist_ok=True)

    # 5. 메모리 효율적 저장 실행 (Main Data)
    main_keys = ["obs", "actions", "dead_mask", "partner_mask", "road_mask"]
    save_large_npz(
        file_paths=trajectory_files, 
        keys_to_process=main_keys, 
        save_path=save_path, 
        save_name=save_name
    )

    # 6. 메모리 효율적 저장 실행 (Global Data)
    global_keys = ["ego_global_rot", "ego_global_pos"]
    save_large_npz(
        file_paths=global_files, 
        keys_to_process=global_keys, 
        save_path=os.path.join(save_path, "global"), 
        save_name=f"global_{save_name}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'])
    args = parser.parse_args()

    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    save_path = "/data/full_version/processed/final_rebuttal/"
    save_name = f"{args.dataset}_{args.num_scene}_seed{args.seed}.npz"
    subset_path = f"/data/full_version/processed/{args.dataset}_subset_v5"

    run(save_path, save_name, subset_path, args.num_scene, seed=args.seed)