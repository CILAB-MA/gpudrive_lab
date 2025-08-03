import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import make_blobs

# 예제 임베딩 생성 (300개, 64차원, 3클래스)
X, y = make_blobs(n_samples=300, n_features=64, centers=3, random_state=42)

# UMAP 임베딩
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# 시각화
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f'Class {label}', alpha=0.7)

plt.title('UMAP Visualization')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.legend()
plt.grid(True)
plt.show()
