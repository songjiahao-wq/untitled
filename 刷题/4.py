import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 假设你的二维数据如下，你可以用你自己的数据替换这个示例数据
data = [
    [1, 2], [1.5, 2.5], [2, 1], [2.5, 1.5], [3, 3], [3.5, 3.5], [4, 2], [4.5, 2.5],
    [10, 9], [10.5, 9.5], [11, 10], [11.5, 10.5], [12, 9], [12.5, 9.5], [13, 10], [13.5, 10.5]
]

data_array = np.array(data)

# 提取x和y坐标
x = data_array[:, 0]
y = data_array[:, 1]

# 计算数据点密度
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# 创建网格数据
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()]))

# 使用 pcolormesh 函数创建密度热力图
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='coolwarm', shading='auto')

# 显示热力图
plt.show()
