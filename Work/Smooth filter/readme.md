# filter
## 实时低通滤波的关键修改
1. 
```python
import numpy as np

class LowPassFilter:
    def __init__(self, alpha=0.3, initial_data=None):
        self.alpha = alpha
        self.prev_filtered = initial_data  # 保存上一帧的滤波结果
    
    def update(self, new_data):
        if self.prev_filtered is None:
            self.prev_filtered = new_data
            return new_data
        filtered = self.alpha * new_data + (1 - self.alpha) * self.prev_filtered
        self.prev_filtered = filtered  # 更新状态
        return filtered
```
2. 调用示例
```python
# 初始化滤波器（每个关键点单独一个滤波器）
num_keypoints = 3  # 假设有3个关键点
filters = [LowPassFilter(alpha=0.3) for _ in range(num_keypoints)]

# 模拟实时数据流（每次输入一帧的关键点数据）
def process_real_time_frame(frame_data):
    filtered_frame = []
    for k in range(num_keypoints):
        # 更新当前关键点的滤波器
        filtered_k = filters[k].update(frame_data[k])
        filtered_frame.append(filtered_k)
    return np.array(filtered_frame)

# 示例：模拟连续3帧输入
frame1 = np.array([[1.2, 2.3, 0.5], [4.5, 1.1, 2.2], [3.3, 0.7, 1.8]])
frame2 = np.array([[1.3, 2.2, 0.6], [4.6, 1.0, 2.3], [3.2, 0.8, 1.9]])
frame3 = np.array([[1.4, 2.1, 0.7], [4.7, 0.9, 2.4], [3.1, 0.9, 2.0]])

# 逐帧处理
print("Filtered Frame1:", process_real_time_frame(frame1))
print("Filtered Frame2:", process_real_time_frame(frame2))
print("Filtered Frame3:", process_real_time_frame(frame3))
```
3. 如果关键点数量多（如25个COCO关键点），可用多线程或向量化加速：
```python
# 向量化版本（所有关键点一起滤波）
class LowPassFilterVectorized:
    def __init__(self, alpha=0.3, initial_data=None):
        self.alpha = alpha
        self.prev_filtered = initial_data
    
    def update(self, new_data):
        if self.prev_filtered is None:
            self.prev_filtered = new_data
            return new_data
        filtered = self.alpha * new_data + (1 - self.alpha) * self.prev_filtered
        self.prev_filtered = filtered
        return filtered

# 初始化（输入整个帧）
filter = LowPassFilterVectorized(alpha=0.3)

# 直接处理完整帧
filtered_frame1 = filter.update(frame1)
filtered_frame2 = filter.update(frame2)
```
4. 异常值处理 在滤波前检测并修正异常跳变：
```python
def update(self, new_data):
    if self.prev_filtered is not None:
        distance = np.linalg.norm(new_data - self.prev_filtered, axis=1)
        mask = distance > 2.0  # 阈值设为2.0米
        new_data[mask] = self.prev_filtered[mask]  # 替换异常值
    # 继续滤波...
```