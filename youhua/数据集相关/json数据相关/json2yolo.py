import os
import json
# 类别映射
class_mapping = {
    "E2": 0,  # 你需要根据你的实际情况进行修改
    "B52": 0,
    "B2": 0,
    "Mirage2000": 0,
    "F4": 0,
    "F14": 0,
    "Tornado": 0,
    "J20": 0,
    "JAS39": 0,
    # 可以根据你的需求添加更多的类别映射
}

# 转换坐标格式为 YOLO 格式
def convert_coordinates(points, img_width, img_height):
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[3][0]), float(points[3][1])

    # 计算中心点和宽高
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    bbox_width = abs(x2 - x1)
    bbox_height = abs(y2 - y1)

    # 将坐标转换为相对于图片宽度和高度的比例
    x_center = x_center / img_width
    y_center = y_center / img_height
    bbox_width = bbox_width / img_width
    bbox_height = bbox_height / img_height

    return x_center, y_center, bbox_width, bbox_height

# 处理整个目录
input_directory = r'D:\xian_yu\code\yolov5-master\dataset\test\json'  # 替换为你的输入目录
output_directory = r'D:\xian_yu\code\yolov5-master\dataset\test\labels'  # 替换为你的输出目录

for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        with open(os.path.join(input_directory, filename)) as f:
            data = json.load(f)
            img_width = int(data['imageWidth'])
            img_height = int(data['imageHeight'])

            # 保存为 YOLO 的 TXT 文件
            with open(os.path.join(output_directory, filename.replace(".json", ".txt")), 'w') as output_file:
                for shape in data['shapes']:
                    label = shape['label']
                    class_id = class_mapping.get(label)
                    if class_id is not None:
                        coordinates = convert_coordinates(shape['points'], img_width, img_height)
                        line = f"{class_id} {' '.join(map(str, coordinates))}"
                        output_file.write(line + "\n")