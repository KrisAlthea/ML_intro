# Hierarchical Clustering
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import numpy as np
#
# # 生成模拟数据
# n_samples = 300
# n_features = 2
# centers = 4
# X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=1.0, random_state=42)
#
#
# # 定义一个函数，用于绘制每次迭代的聚类结果
# def plot_clusters(X, centroids, labels, n_cluster, iteration, ax):
#     # 绘制样本点
#     plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
#
#     # 绘制质心
#     plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.7, marker='o')
#     plt.title(f"Iteration {iteration}")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#
#
# # 运行K-均值聚类，并在每次迭代后绘制结果
# kmeans = KMeans(n_clusters=centers, random_state=42, n_init=1, max_iter=1, init='random')
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
# for i in range(6):
#     # 单步执行K-均值算法
#     kmeans.max_iter = i + 1
#     kmeans.fit(X)
#     centroids = kmeans.cluster_centers_
#     labels = kmeans.labels_
#
#     # 绘制当前迭代的聚类结果
#     ax = axs[i // 3, i % 3]
#     plt.sca(ax)
#     plot_clusters(X, centroids, labels, centers, i + 1, ax)
#
# plt.tight_layout()
# plt.show()


# Spectral Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# 生成模拟数据
X, y = make_moons(n_samples=50, noise=0.08, random_state=42)

# 应用谱聚类
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = sc.fit_predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title("Spectral Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
