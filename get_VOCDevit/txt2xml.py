import os
import glob
from PIL import Image

voc_annotations = r'D:\xian_yu\xianyun_lunwen\吸烟和打手机识别\大论文抽打数据集\final_data_small\test/annotations/'
yolo_txt = r'D:\xian_yu\xianyun_lunwen\吸烟和打手机识别\大论文抽打数据集\final_data_small\test/labels/'
img_path = r'D:\xian_yu\xianyun_lunwen\吸烟和打手机识别\大论文抽打数据集\final_data_small\test/images/'
labels = ['A', 'B', 'C']  # label for datasets
# 图像存储位置
src_img_dir = img_path  # 添加你的路径
# 图像的txt文件存放位置


src_txt_dir = yolo_txt
src_xml_dir = voc_annotations

img_Lists = glob.glob(src_img_dir + '/*.jpg')

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))

img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

for img in img_names:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    # 打开txt文件
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    print(gt)
    if gt:
        # 将主干部分写入xml文件中
        xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for img_each_label in gt:
            spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            print(f'spt:{spt}')
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + str(labels[int(spt[0])]) + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')

            center_x = round(float(spt[1].strip()) * width)
            center_y = round(float(spt[2].strip()) * height)
            bbox_width = round(float(spt[3].strip()) * width)
            bbox_height = round(float(spt[4].strip()) * height)
            xmin = str(int(center_x - bbox_width / 2))
            ymin = str(int(center_y - bbox_height / 2))
            xmax = str(int(center_x + bbox_width / 2))
            ymax = str(int(center_y + bbox_height / 2))

            xml_file.write('            <xmin>' + xmin + '</xmin>\n')
            xml_file.write('            <ymin>' + ymin + '</ymin>\n')
            xml_file.write('            <xmax>' + xmax + '</xmax>\n')
            xml_file.write('            <ymax>' + ymax + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')

        xml_file.write('</annotation>')


