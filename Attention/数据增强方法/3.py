import json

# 读取COCO128的annotations文件
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# 创建一个train.txt文件，并将图像文件名和标注信息写入其中
with open('train.txt', 'w') as f:
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']
        x, y, w, h = bbox
        label = ann['category_id']

        # 将图像文件名和标注信息写入train.txt文件中
        f.write(f'path/to/images/{img_id}.jpg,{x},{y},{w},{h},{label}\n')
