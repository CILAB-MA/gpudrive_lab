import os
import numpy as np
from tqdm import tqdm
import gc
import random


def extract_scene_id(filename):
    return int(filename.split("_")[1].split(".")[0])


def compute_scene_unit(scene_ids):
    """
    파일명 기준으로 scene 단위를 계산.
    예: 78500, 78600 → 100, 79000 → 200 등
    전체 ID 차이 중 최소값을 사용 (100/200 섞여 있어도 안정적)
    """
    diffs = [scene_ids[i+1] - scene_ids[i] for i in range(len(scene_ids)-1)]
    return min(diffs)


def run(save_path, save_name, subset_path, num_scenes, seed=42):
    ### ------------------------------------------------------
    ### 1. subset 파일에서 scene_id만 추출
    ### ------------------------------------------------------
    all_files = [f for f in os.listdir(subset_path) if f.endswith(".npz") and f.startswith("trajectory_")]
    scene_ids = sorted([extract_scene_id(f) for f in all_files])

    if len(scene_ids) < 2:
        raise ValueError("파일이 2개 미만이면 scene 단위를 계산할 수 없음.")

    ### ------------------------------------------------------
    ### 2. scene 단위 계산 (100 or 200)
    ### ------------------------------------------------------
    scene_unit = compute_scene_unit(scene_ids)
    print(f"[INFO] scene_unit = {scene_unit}")

    ### ------------------------------------------------------
    ### 3. 필요한 파일 수 계산
    ### ------------------------------------------------------
    need_files = (num_scenes + scene_unit - 1) // scene_unit   # 올림
    print(f"[INFO] 필요한 파일 개수 = {need_files}")

    if need_files > len(scene_ids):
        raise ValueError("요청한 num_scenes를 만족할 만큼 파일이 부족함.")

    ### ------------------------------------------------------
    ### 4. scene_ids에서 랜덤 샘플링
    ### ------------------------------------------------------
    random.seed(seed)
    chosen_ids = random.sample(scene_ids, need_files)
    chosen_ids = sorted(chosen_ids)

    chosen_files = [f"trajectory_{sid}.npz" for sid in chosen_ids]
    print(f"[INFO] 선택된 파일 수 = {len(chosen_files)}")
    print(f"[INFO] 선택된 파일들 = {chosen_files}")

    ### ------------------------------------------------------
    ### 5. 데이터 로딩 및 concat
    ### ------------------------------------------------------
    obs_list = []
    actions_list = []
    dead_masks_list = []
    partner_masks_list = []
    road_masks_list = []
    ego_global_rots_list = []
    ego_global_poss_list = []

    total_loaded = 0

    for filename in tqdm(chosen_files):
        # subset 파일 로딩
        file_path = os.path.join(subset_path, filename)
        data = np.load(file_path)

        obs_list.append(data["obs"])
        actions_list.append(data["actions"])
        dead_masks_list.append(data["dead_mask"])
        partner_masks_list.append(data["partner_mask"])
        road_masks_list.append(data["road_mask"])

        scene_count = data["obs"].shape[0]
        total_loaded += scene_count

        del data
        gc.collect()

        # global 파일 로딩
        global_file = f"global_{filename}"
        global_path = os.path.join(subset_path, "global", global_file)

        g = np.load(global_path)
        ego_global_rots_list.append(g["ego_global_rot"])
        ego_global_poss_list.append(g["ego_global_pos"])
        del g
        gc.collect()

    print(f"[INFO] 로딩된 총 scene 수 = {total_loaded} → 최종 {num_scenes}개만 사용")

    ### ------------------------------------------------------
    ### 6. concat 및 num_scenes만큼 슬라이싱
    ### ------------------------------------------------------
    obs = np.concatenate(obs_list, axis=0)[:num_scenes]
    actions = np.concatenate(actions_list, axis=0)[:num_scenes]
    dead_masks = np.concatenate(dead_masks_list, axis=0)[:num_scenes]
    partner_masks = np.concatenate(partner_masks_list, axis=0)[:num_scenes]
    road_masks = np.concatenate(road_masks_list, axis=0)[:num_scenes]

    ego_global_rots = np.concatenate(ego_global_rots_list, axis=0)[:num_scenes]
    ego_global_poss = np.concatenate(ego_global_poss_list, axis=0)[:num_scenes]

    ### ------------------------------------------------------
    ### 7. 저장
    ### ------------------------------------------------------
    print("[INFO] compressing & saving...")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "global"), exist_ok=True)

    np.savez_compressed(
        os.path.join(save_path, save_name),
        obs=obs,
        actions=actions,
        dead_mask=dead_masks,
        partner_mask=partner_masks,
        road_mask=road_masks,
    )

    np.savez_compressed(
        os.path.join(save_path, "global", f"global_{save_name}"),
        ego_global_rot=ego_global_rots,
        ego_global_pos=ego_global_poss,
    )

    print("[INFO] done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scene', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'])
    args = parser.parse_args()

    save_path = "/data/full_version/processed/test/"
    save_name = f"{args.dataset}_{args.num_scene}.npz"
    subset_path = f"/data/full_version/processed/{args.dataset}_subset_v5"

    run(save_path, save_name, subset_path, args.num_scene, seed=args.seed)
