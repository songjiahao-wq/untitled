import os

def rename_images(directory):
    # 获取文件夹中所有的.jpg文件
    files = [file for file in os.listdir(directory) if file.endswith('.jpg')]
    files.sort()  # 如果需要，可以根据需要排序文件

    # 重命名文件
    for i, file in enumerate(files):
        new_filename = f"{i+1:06}.jpg"  # 生成新的文件名，如 000001.jpg, 000002.jpg, 等
        os.rename(os.path.join(directory, file), os.path.join(directory, new_filename))
        print(f"Renamed {file} to {new_filename}")

# 使用示例
directory = r'D:\xian_yu\daipao\2\mobilenets-ssd-pytorch-master\data\VOCdevkit\VOC2007\JPEGImages'  # 将此路径替换为你的图片文件夹的路径
rename_images(directory)
