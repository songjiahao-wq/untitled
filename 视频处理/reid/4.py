import os
import shutil
import random

# 输入数据文件夹
input_pedestrian_folder = r"F:\sjh\code\deep-person-reid\reid-data\my_data\person"
input_vehicle_folder = r"F:\sjh\code\deep-person-reid\reid-data\my_data\car"

# 输出数据文件夹
output_root = r"F:\sjh\code\deep-person-reid\reid-data\my_data/Market1501"
bounding_box_train = os.path.join(output_root, "bounding_box_train")
bounding_box_test = os.path.join(output_root, "bounding_box_test")
query = os.path.join(output_root, "query")

if os.path.exists(bounding_box_train):

    shutil.rmtree(bounding_box_train)
    shutil.rmtree(bounding_box_test)
    shutil.rmtree(query)
os.makedirs(bounding_box_train, exist_ok=True)
os.makedirs(bounding_box_test, exist_ok=True)
os.makedirs(query, exist_ok=True)

def rename_and_split_images(input_folder, pid_start, camid, query_bool=False,classes=0):
    pid = pid_start
    image_files = os.listdir(input_folder)
    frame = 1
    for image_file in image_files:
        # 为每个行人/车辆分配一个唯一的pid
        # pid += 1
        # if classes == 1:
        #     pid = 1
        # elif classes == 2:
        #     pid = 2
        src_path = os.path.join(input_folder, image_file)

        # 在这里，我们假设每个输入文件夹只有一个摄像头和连续的帧

        frame += 1
        new_image_name = f"{pid:04d}_c{camid}s{camid}_{frame:06d}.jpg"
        dest_path = os.path.join(bounding_box_train, new_image_name)
        shutil.copyfile(src_path, dest_path)

        # 随机选择一些图像分配给bounding_box_test和query
        if random.random() < 0.1:
            frame += 1

            new_image_name = f"{pid:04d}_c{camid}s{camid}_{frame:06d}.jpg"
            dest_path = os.path.join(bounding_box_test, new_image_name)
            shutil.copyfile(src_path, dest_path)

            frame += 1
            if query_bool:
                camid=2
            new_image_name = f"{pid:04d}_c{camid}s{camid}_{frame:06d}.jpg"
            dest_path = os.path.join(query, new_image_name)
            shutil.copyfile(src_path, dest_path)

    return pid

# 处理行人数据
last_pid = rename_and_split_images(input_pedestrian_folder, 0, 1,query_bool=True,classes=1)

# 处理车辆数据
rename_and_split_images(input_vehicle_folder, 1, 2, query_bool=True,classes=2)
