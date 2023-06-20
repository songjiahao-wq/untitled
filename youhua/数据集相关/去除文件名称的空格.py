import os
import shutil

# 设置目标目录
dir_path = r'D:\my_job\MY_Github\faster-rcnn-pytorch-master\VOCdevkit\VOC2007\JPEGImages'

# 遍历目录中的文件
for filename in os.listdir(dir_path):
    # 仅考虑xml文件
    if filename.endswith('.jpg'):
        # 创建没有空格的新文件名
        new_filename = filename.replace(' ', '')
        # 获取文件的原始和新路径
        original_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, new_filename)
        # 重命名文件
        shutil.move(original_path, new_path)
