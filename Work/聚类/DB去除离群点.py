import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 构造 3D 平面点（Z = 0.5）
np.random.seed(42)
plane_points = np.random.rand(15, 2)
plane_points = np.hstack((plane_points, np.full((15, 1), 0.5)))  # Z = 0.5

# 2. 添加一个“飘”的点（离群）
outlier_point = np.array([[0.5, 0.5, 2.0]])  # 明显偏离平面

# 3. 合并数据
points_3d = np.vstack((plane_points, outlier_point))

# 4. 使用 DBSCAN 聚类检测离群点
clustering = DBSCAN(eps=0.2, min_samples=3).fit(points_3d)
labels = clustering.labels_

# -1 表示离群点
inlier_mask = labels != -1
outlier_mask = labels == -1

# 5. 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[inlier_mask][:, 0], points_3d[inlier_mask][:, 1], points_3d[inlier_mask][:, 2],
           c='b', label='Inliers')
ax.scatter(points_3d[outlier_mask][:, 0], points_3d[outlier_mask][:, 1], points_3d[outlier_mask][:, 2],
           c='r', label='Outlier', s=100)
ax.set_title("3D Points with Outlier Detection via DBSCAN")
ax.legend()
plt.show()
