# -*- coding: utf-8 -*-
# @Time    : 2024/8/2 14:14
# @Author  : sjh
# @Site    : 
# @File    : OneEuroFilter.py
# @Comment :
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Adapted from https://github.com/HoBeom/OneEuroFilter-Numpy
# Original licence: Copyright (c)  HoBeom Jeon, under the MIT License.
# ------------------------------------------------------------------------------
import warnings
from time import time

import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:

    def __init__(self,
                 x0,
                 dx0=0.0,
                 min_cutoff=1.7,
                 beta=0.3,
                 d_cutoff=30.0,
                 fps=None):
        """One Euro Filter for keypoints smoothing.

        Args:
            x0 (np.ndarray[K, 2]): Initialize keypoints value
            dx0 (float): 0.0
            min_cutoff (float): parameter for one euro filter
            beta (float): parameter for one euro filter
            d_cutoff (float): Input data FPS
            fps (float): Video FPS for video inference
        """
        warnings.warn(
            'OneEuroFilter from '
            '`mmpose/core/post_processing/one_euro_filter.py` will '
            'be deprecated in the future. Please use Smoother'
            '(`mmpose/core/post_processing/smoother.py`) with '
            'OneEuroFilter (`mmpose/core/post_processing/temporal_'
            'filters/one_euro_filter.py`).', DeprecationWarning)

        # The parameters.
        self.data_shape = x0.shape
        self.min_cutoff = np.full(x0.shape, min_cutoff)
        self.beta = np.full(x0.shape, beta)
        self.d_cutoff = np.full(x0.shape, d_cutoff)
        # Previous values.
        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.full(x0.shape, dx0)
        self.mask_prev = np.ma.masked_where(x0 <= 0, x0)
        self.realtime = True
        if fps is None:
            # Using in realtime inference
            self.t_e = None
            self.skip_frame_factor = d_cutoff
            self.fps = d_cutoff
        else:
            # fps using video inference
            self.realtime = False
            self.fps = float(fps)
            self.d_cutoff = np.full(x0.shape, self.fps)

        self.t_prev = time()

    def __call__(self, x, t_e=1.0):
        """Compute the filtered signal.

        Hyper-parameters (cutoff, beta) are from `VNect
        <http://gvv.mpi-inf.mpg.de/projects/VNect/>`__ .

        Realtime Camera fps (d_cutoff) default 30.0

        Args:
            x (np.ndarray[K, 2]): keypoints results in frame
            t_e (Optional): video skip frame count for posetrack
                evaluation
        """
        assert x.shape == self.data_shape

        t = 0
        if self.realtime:
            t = time()
            t_e = (t - self.t_prev) * self.skip_frame_factor
        t_e = np.full(x.shape, t_e)

        # missing keypoints mask
        mask = np.ma.masked_where(x <= 0, x)

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e / self.fps, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e / self.fps, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # missing keypoints remove
        np.copyto(x_hat, -10, where=mask.mask)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        self.mask_prev = mask

        return x_hat

# 示例数据：带噪声的正弦波信号
np.random.seed(42)
t = np.linspace(0, 10, 500)
true_signal = np.sin(t)
noisy_signal = true_signal + np.random.normal(scale=0.5, size=t.shape)

# 设置初始参数
te = t[1] - t[0]  # 采样时间间隔
initial_value = np.array([[noisy_signal[0]]])  # 传入初始值

# 创建OneEuroFilter实例
filter = OneEuroFilter(x0=initial_value, fps=1/te)

# 对信号进行滤波
filtered_signal = np.array([filter(np.array([[x]]), te)[0, 0] for x in noisy_signal])

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