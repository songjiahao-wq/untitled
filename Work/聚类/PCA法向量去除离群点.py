# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 下午4:55
# @Author  : sjh
# @Site    : 
# @File    : 333.py
# @Comment :
import numpy as np
from sklearn.neighbors import NearestNeighbors

def remove_outliers_by_normal_consistency(points: np.ndarray, k: int = 6, cos_thresh: float = 0.9):
    """
    移除3D点集中法向量方向与主方向不一致的离群点

    Args:
        points (np.ndarray): 输入点 (N, 3)
        k (int): 邻域大小，用于PCA估计局部法向量
        cos_thresh (float): 与平均法向量余弦相似度的阈值，小于该值为离群点

    Returns:
        inlier_points (np.ndarray): 去除离群点后的点集
        mask (np.ndarray): bool数组，标记每个点是否是内点
    """
    if len(points) < k:
        raise ValueError("点数量必须大于邻域大小 k")

    normals = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)

    for i in range(len(points)):
        _, indices = nbrs.kneighbors([points[i]])
        neighbors = points[indices[0]]
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(neighbors_centered.T)
        _, _, vh = np.linalg.svd(cov)
        normal = vh[-1]  # 法向量（最小特征值方向）
        normals.append(normal)

    normals = np.array(normals)
    avg_normal = np.mean(normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal)

    cos_similarities = np.abs(normals @ avg_normal)
    mask = cos_similarities >= cos_thresh
    inlier_points = points[mask]
    return inlier_points, mask
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time
    # 构造测试数据
    plane_pts = np.random.rand(15, 2)
    plane_pts = np.hstack((plane_pts, np.full((15, 1), 0.5)))  # Z平面
    # 添加一个“轻微”离群点（稍微偏离平面）
    outlier = np.array([[0.5, 0.5, 0.8]])
    points = np.vstack((plane_pts, outlier))
    t1 = time.time()
    filtered_points, mask = remove_outliers_by_normal_consistency(points)
    print("去除离群点耗时:", time.time() - t1)
    # 可视化前后对比
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("原始点集")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("去除离群点后")
    ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], c='b')

    plt.show()
