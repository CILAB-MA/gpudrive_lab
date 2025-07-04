import numpy as np
import os
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
# 예: questions shape (N, D), answers shape (N, A)
# 예시 데이터 로드
data_path = '/data/full_version/processed/final/reasoning'
data_file = "reasoning_validation_trajectory_10000.npz"
exp_name = 'ego'

with np.load(os.path.join(data_path, data_file), mmap_mode='r') as npz:
    questions = npz[f'{exp_name}_qs']  # (N, D)
    answers = npz[f'{exp_name}_as']    # (N, A)
    masks = npz[f'{exp_name}_masks']

concat_vecs = np.concatenate([questions, answers], axis=-1)
flat_vecs = concat_vecs.reshape(-1, 768)
_, unique_indices = np.unique(flat_vecs, axis=0, return_index=True)
unique_mask_flat = np.zeros(flat_vecs.shape[0], dtype=bool)
unique_mask_flat[unique_indices] = True
unique_mask = unique_mask_flat.reshape(1281, 20)
final_mask = ~((unique_mask == True) & (masks == False))
masked_questions_sum = questions[~final_mask].sum(-1)
masked_answers = answers[~final_mask]
grouped_answers = defaultdict(list)

for q, a in zip(masked_questions_sum, masked_answers):
    q_key = str(q)
    grouped_answers[q_key].append(a)

# 그룹 내 cosine similarity 저장
all_group_cos_sims = []
exact_one_count = 0
total_pairs = 0
for q_key, ans_list in grouped_answers.items():
    ans_arr = np.stack(ans_list, axis=0)
    n = ans_arr.shape[0]
    if n > 1:
        for i, j in combinations(range(n), 2):
            v1, v2 = ans_arr[i], ans_arr[j]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            all_group_cos_sims.append(cos_sim)
            total_pairs += 1
            if np.isclose(cos_sim, 1.0, atol=1e-6):
                exact_one_count += 1
# numpy array 변환
all_group_cos_sims = np.array(all_group_cos_sims)
print(f"Total pairs checked: {total_pairs}")
print(f"Number of exactly 1.0 cosine similarity pairs: {exact_one_count}")
# 통계 출력
print(f"Group 내부 cosine similarity 평균: {np.mean(all_group_cos_sims):.4f}")
print(f"Group 내부 cosine similarity 표준편차: {np.std(all_group_cos_sims):.4f}")
print(f"최소값: {np.min(all_group_cos_sims):.4f}, 최대값: {np.max(all_group_cos_sims):.4f}")

# histogram 그리기
plt.figure(figsize=(8, 5))
plt.hist(all_group_cos_sims, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Cosine Similarity (within same question group)")
plt.ylabel("Frequency")
plt.title("Distribution of Cosine Similarity within Question Groups")
plt.grid(True)
plt.savefig(f'{exp_name}_cossim.png')
