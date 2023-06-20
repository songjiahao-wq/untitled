import glob
import os

# 获取当前目录下所有的txt文件
txt_files = glob.glob(r"D:\xian_yu\xianyun_lunwen\xiaci2000\datasets\labels\val/*.txt")

for txt_file in txt_files:
    # 读取文件中的所有行
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 只保留不以2开头的行
    new_lines = [line for line in lines if not line.startswith('1')]

    # 将结果写回文件
    with open(txt_file, 'w') as f:
        f.writelines(new_lines)
