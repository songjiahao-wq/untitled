import os

# 指定要删除文件的目录路径
directories = ['./data/images/train', './data/labels/train', './data/images/val', './data/labels/val']

# 遍历每个目录并删除其中的文件
for directory in directories:
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

