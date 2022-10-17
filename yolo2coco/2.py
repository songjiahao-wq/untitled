import json
import os
import cv2

#-------------------可用-----------------------------------
'''
data:2022.4.6
check:Y
function:可将yolo格式的数据集转换为coco格式的(2)

需要准备：
classes.txt：一行就是一个类，不需要数字，只要类名称
annos.txt：由上一个.py文件生成
images：与annos.txt对应的图片，需要有序号

生成.json文件，在annotations文件下
'''

# ------------用os提取images文件夹中的图片名称，并且将BBox都读进去------------
# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),
# 以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = r'D:/songjiahao/DATA/Crowdhuman100/'
# 用于创建训练集或验证集
phase = 'val'  # 需要修正，保存后的json文件名

# dataset用于保存所有数据的图片信息和标注信息
dataset = {'categories': [], 'annotations': [], 'images': []}

# 打开类别标签
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
indexes = os.listdir(os.path.join(root_path, 'valid/images'))

# 统计处理图片的数量
global count
count = 0

# 读取Bbox信息
with open(os.path.join(root_path, 'valid/annos.txt')) as tr:
    annos = tr.readlines()

    # ---------------接着将，以上数据转换为COCO所需要的格式---------------
    for k, index in enumerate(indexes):
        count += 1
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(os.path.join(root_path, './valid/images/') + index)
        height, width, _ = im.shape

        # 添加图像的信息到dataset中
        print(index)
        dataset['images'].append({'file_name': index.replace("\\", "/"),
                                  'id': int(index[0:5]),  # 提取文件名 里的数字标号 必须是int类型，不能是str
                                  'width': width,
                                  'height': height})

        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

            # 如果图像的名称和标记的名称对上，则添加标记
            if parts[0] == index:
                # 类别
                cls_id = parts[1]
                # x_min
                x1 = float(parts[2])
                # y_min
                y1 = float(parts[3])
                # x_max
                x2 = float(parts[4])
                # y_max
                y2 = float(parts[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': ii,
                    'image_id': int(index[0:5]),	# 提取文件名里的数字标号  必须是int类型，不能是str
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

        print('{} images handled'.format(count))

# 保存结果的文件夹
folder = os.path.join(root_path, './valid/annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, './valid/annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=1)
