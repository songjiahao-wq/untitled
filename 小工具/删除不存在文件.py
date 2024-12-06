# -*- coding: utf-8 -*-
# @Time    : 2024/11/6 16:50
# @Author  : sjh
# @Site    : 
# @File    : 删除不存在文件.py
# @Comment :
import os
import glob

# 目录路径
dir_a = r"D:\BaiduSyncdisk\work\yolov5_tensorRT\yolov5-master-mutiltask\resize\color_selected"  # A目录路径
dir_b = r"D:\BaiduSyncdisk\work\yolov5_tensorRT\yolov5-master-mutiltask\resize\depth"  # B目录路径

# 获取目录B中的文件名，不包括后缀
a_files_no_ext = {os.path.splitext(file)[0] for file in os.listdir(dir_a)}

# 遍历目录A中的jpg文件
for file_path in glob.glob(os.path.join(dir_b, "*.png")):
    # 获取文件名（不包括路径和后缀）
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]

    # 判断目录B中是否有同名文件
    if file_name_no_ext not in a_files_no_ext:
        # 如果B中没有同名文件，则删除A目录中的文件
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"Kept: {file_path}")
