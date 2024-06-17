import glob
import os
from tqdm import tqdm
# 获取当前目录下所有的txt文件
txt_files = glob.glob(r"E:\Download\Datasets\football\Football Analyzer.v1i.yolov8\train\labels/*.txt")

for txt_file in tqdm(txt_files):

    # 读取文件中的所有行
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 只保留不以2开头的行
    # new_lines = [line for line in lines if not line.startswith('1')]
    new_lines = [line for line in lines if line.startswith('0')]

    # 将结果写回文件
    with open(txt_file, 'w') as f:
        f.writelines(new_lines)
