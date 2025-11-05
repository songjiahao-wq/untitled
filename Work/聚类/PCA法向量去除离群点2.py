# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 下午4:51
# @Author  : sjh
# @Site    : 
# @File    : 222.py
# @Comment :
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 1. 构造平面上的点 (Z = 0.5)，加一点离群
np.random.seed(42)
plane_pts = np.random.rand(15, 2)
plane_pts = np.hstack((plane_pts, np.full((15, 1), 0.5)))  # Z=0.5
outlier = np.array([[0.5, 0.5, 0.8]])
points = np.vstack((plane_pts, outlier))

# 2. 用 PCA 分析每个点的局部法向量
def estimate_normals_pca(points, k=6):
    normals = []
    for i in range(len(points)):
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = nbrs.kneighbors([points[i]])
        neighbors = points[indices[0]]
        neighbors_centered = neighbors - neighbors.mean(axis=0)

        # PCA: 取最小特征值对应的特征向量作为法向量
        cov = np.cov(neighbors_centered.T)
        _, _, vh = np.linalg.svd(cov)
        normal = vh[-1]  # 主轴方向（最小方差）
        normals.append(normal)
    return np.array(normals)
t1 = time.time()
normals = estimate_normals_pca(points)
print("计算法向量耗时:", time.time() - t1)

# 3. 计算法向量与平均法向量的夹角（余弦）
avg_normal = np.mean(normals[:-1], axis=0)
avg_normal /= np.linalg.norm(avg_normal)

cos_similarities = np.abs(np.dot(normals, avg_normal))  # 越接近1越一致
is_outlier = cos_similarities < 0.9

# 4. 可视化点与法向量
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 显示点
for i, pt in enumerate(points):
    color = 'r' if is_outlier[i] else 'b'
    ax.scatter(*pt, c=color)
    ax.quiver(pt[0], pt[1], pt[2],
              normals[i][0], normals[i][1], normals[i][2],
              length=0.2, normalize=True, color=color)

ax.set_title("PCA Estimated Normals (Red = Outlier)")
plt.show()
