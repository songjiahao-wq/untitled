import os
import shutil

def process_images(src_folder, dst_folder, person_id_start,query=False,test=False):
    for person_id, image_name in enumerate(sorted(os.listdir(src_folder)), start=person_id_start):
        person_id_tmp =person_id
        person_id = 1
        src_image_path = os.path.join(src_folder, image_name)
        dst_image_name = f"{person_id:04d}_c1s1_{person_id_tmp + 1:04d}_00.jpg"
        if query:
            dst_image_name = f"{person_id:04d}_c2s2_{person_id_tmp+1:04d}_00.jpg"
        if test:
            dst_image_name = f"{person_id:04d}_c3s3_{person_id_tmp + 1:04d}_00.jpg"
        dst_image_path = os.path.join(dst_folder, dst_image_name)
        shutil.copy(src_image_path, dst_image_path)
folder = r"F:\sjh\code\deep-person-reid\reid-data\my_data/"
raw_train_folder =folder+ "raw_train"
raw_test_folder = folder+ "raw_test"
raw_query_folder =folder+ "raw_query"

output_folder = folder+"Market-1501"
bounding_box_train_folder = os.path.join(output_folder, "bounding_box_train")
bounding_box_test_folder = os.path.join(output_folder, "bounding_box_test")
query_folder = os.path.join(output_folder, "query")

os.makedirs(bounding_box_train_folder, exist_ok=True)
os.makedirs(bounding_box_test_folder, exist_ok=True)
os.makedirs(query_folder, exist_ok=True)

# 处理训练集照片
process_images(raw_train_folder, bounding_box_train_folder, person_id_start=0)

# 处理测试集照片
process_images(raw_test_folder, bounding_box_test_folder, person_id_start=1000,test=True)

# 处理查询集照片
process_images(raw_query_folder, query_folder, person_id_start=2000,query=True)
