#第一天学习
#判断同一个照片和labels是否都存在
import os

# 图片和标签文件夹路径
img_path = r'D:\BaiduSyncdisk\work\YOLO\YOLOv8_trt\ultralytics\runs\detect\predict\train2017'
labels_path = r'D:\BaiduSyncdisk\work\YOLO\YOLOv8_trt\ultralytics\runs\detect\predict\labels'

# 获取图片文件夹中所有文件名
img_dir = os.listdir(img_path)

# 遍历图片文件夹中的文件
for img_file in img_dir:
    # 构造对应的标签文件名
    label_name = os.path.splitext(img_file)[0] + '.txt'

    # 获取标签文件夹中所有文件名
    label_dir = os.listdir(labels_path)

    # 检查对应的标签文件是否存在
    if label_name not in label_dir:
        print(f"标签文件 {label_name} 不存在，将删除图片文件 {img_file}")

        # 构造完整的图片文件路径
        img_file_path = os.path.join(img_path, img_file)

        # 删除图片文件
        os.remove(img_file_path)
        print(f"图片文件 {img_file} 已被删除。")