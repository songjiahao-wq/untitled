# -*- coding: utf-8 -*-
# @Time    : 2025/2/9 19:46
# @Author  : sjh
# @Site    : 
# @File    : 处理数据集.py
# @Comment :
import os
import random
import shutil
data_dir = "D:/BaiduNetdiskDownload/crowdhuman/crow"

def filter_head_labels(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            label_path = os.path.join(data_dir, file_name)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 只保留类别 ID 为 1 的行（人头）
            head_lines = [line for line in lines if line.startswith('0 ')]

            # 写回文件
            with open(label_path, 'w') as f:
                f.writelines(head_lines)


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 获取所有文件名（不带扩展名）
    files = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.jpg')]

    # 随机打乱
    random.shuffle(files)

    # 计算划分点
    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 划分数据集
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # 创建输出目录
    output_dir = os.path.join(data_dir, "split")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # 复制文件到相应的目录
    for file in train_files:
        shutil.copy(os.path.join(data_dir, file + ".jpg"), os.path.join(output_dir, "train", file + ".jpg"))
        shutil.copy(os.path.join(data_dir, file + ".txt"), os.path.join(output_dir, "train", file + ".txt"))

    for file in val_files:
        shutil.copy(os.path.join(data_dir, file + ".jpg"), os.path.join(output_dir, "val", file + ".jpg"))
        shutil.copy(os.path.join(data_dir, file + ".txt"), os.path.join(output_dir, "val", file + ".txt"))

    for file in test_files:
        shutil.copy(os.path.join(data_dir, file + ".jpg"), os.path.join(output_dir, "test", file + ".jpg"))
        shutil.copy(os.path.join(data_dir, file + ".txt"), os.path.join(output_dir, "test", file + ".txt"))


if __name__ == "__main__":
    # 过滤标注，只保留人头
    filter_head_labels(data_dir)

    # 划分数据集
    split_dataset(data_dir)