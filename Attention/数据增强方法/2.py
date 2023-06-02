import os

folder_path = r'D:\yanyi\project_process\datasets\coco128\images\train2017'
file_names = os.listdir(folder_path)

with open('train.txt', 'w') as f:
    for file_name in file_names:
        if file_name.endswith('.jpg'):
            f.write(os.path.join(folder_path, file_name) + '\n')