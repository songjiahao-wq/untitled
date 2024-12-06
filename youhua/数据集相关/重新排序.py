import os
import shutil

# 设置目标文件夹路径
folder_path = r"E:\project2024\KinectV2\KinectV2\Chessboard\IR"

# 获取所有 .jpg 文件
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 按照文件名排序（如果文件名有数字或日期等，按字母或数字顺序）
jpg_files.sort()

# 对文件进行重新命名
for i, file_name in enumerate(jpg_files):
    i = i + 0
    # 构造新文件名，重新从 0 开始
    new_name = f"{i}.jpg"

    # 获取文件的完整路径
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)

    # 重命名文件
    os.rename(old_path, new_path)

    print(f"Renamed: {file_name} -> {new_name}")
