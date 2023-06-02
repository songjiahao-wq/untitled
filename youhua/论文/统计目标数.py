import os
import glob

# 设置阈值以分类大，中，小目标
small_threshold = 0.02
large_threshold = 0.05

# 计数器
small_count = 0
medium_count = 0
large_count = 0

# 获取所有的标签文件
label_files = glob.glob(r'F:\sjh\DATA\wider+crowd\labels\train/*.txt')

for file in label_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, x, y, w, h = map(float, line.strip().split())
            # 使用宽度和高度的平均值作为目标大小的度量
            avg_dim = (w + h) / 2.0
            if avg_dim < small_threshold:
                small_count += 1
            elif avg_dim < large_threshold:
                medium_count += 1
            else:
                large_count += 1

print(f'Small objects: {small_count}')
print(f'Medium objects: {medium_count}')
print(f'Large objects: {large_count}')
