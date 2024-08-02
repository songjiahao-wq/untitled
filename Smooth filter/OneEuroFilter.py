# -*- coding: utf-8 -*-
# @Time    : 2024/8/2 14:14
# @Author  : sjh
# @Site    : 
# @File    : OneEuroFilter.py
# @Comment :
import numpy as np
import matplotlib.pyplot as plt

class OneEuroFilter:
    def __init__(self, te, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.x = None
        self.dx = 0
        self.te = te
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.alpha = self._alpha(self.mincutoff)
        self.dalpha = self._alpha(self.dcutoff)

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self.te)

    def predict(self, x, te):
        result = x
        if self.x is not None:
            edx = (x - self.x) / te
            self.dx = self.dx + (self.dalpha * (edx - self.dx))
            cutoff = self.mincutoff + self.beta * abs(self.dx)
            self.alpha = self._alpha(cutoff)
            result = self.x + self.alpha * (x - self.x)
        self.x = result
        return result

# 示例数据：带噪声的正弦波信号
np.random.seed(42)
t = np.linspace(0, 10, 500)
true_signal = np.sin(t)
noisy_signal = true_signal + np.random.normal(scale=0.5, size=t.shape)

# 创建OneEuroFilter实例
te = t[1] - t[0]  # 采样时间间隔
filter = OneEuroFilter(te)

# 对信号进行滤波
filtered_signal = np.array([filter.predict(x, te) for x in noisy_signal])

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(t, true_signal, label='True Signal', color='g', linewidth=2)
plt.plot(t, noisy_signal, label='Noisy Signal', color='r', linestyle='--', alpha=0.6)
plt.plot(t, filtered_signal, label='Filtered Signal', color='b', linewidth=2)
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('OneEuroFilter Signal Filtering')
plt.show()
