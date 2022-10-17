import os
from PIL import Image
import glob

yolo_img = 'D:/shuichi_test/yolo/img/'
yolo_txt = 'D:/shuichi_test/yolo/txt/'
voc_xml = 'D:/shuichi_test/voc/annotations/'

# 目标类别
labels = ['HLB', 'health', 'ill']
# 匹配文件路径下的所有jpg文件，并返回列表
img_glob = glob.glob(yolo_img + '*.jpg')

img_base_names = []

for img in img_glob:
    # os.path.basename:取文件的后缀名
    img_base_names.append(os.path.basename(img))

img_pre_name = []

for img in img_base_names:
    # os.path.splitext:将文件按照后缀切分为两块
    temp1, temp2 = os.path.splitext(img)
    img_pre_name.append(temp1)
    print(f'imgpre:{img_pre_name}')
for img in img_pre_name:
    with open(voc_xml + img + '.xml', 'w') as xml_files:
        image = Image.open(yolo_img + img + '.jpg')
        img_w, img_h = image.size
        xml_files.write('<annotation>\n')
        xml_files.write('   <folder>folder</folder>\n')
        xml_files.write(f'   <filename>{img}.jpg</filename>\n')
        xml_files.write('   <source>\n')
        xml_files.write('   <database>Unknown</database>\n')
        xml_files.write('   </source>\n')
        xml_files.write('   <size>\n')
        xml_files.write(f'     <width>{img_w}</width>\n')
        xml_files.write(f'     <height>{img_h}</height>\n')
        xml_files.write(f'     <depth>3</depth>\n')
        xml_files.write('   </size>\n')
        xml_files.write('   <segmented>0</segmented>\n')
        with open(yolo_txt + img + '.txt', 'r') as f:
            # 以列表形式返回每一行
            lines = f.read().splitlines()
            for each_line in lines:
                line = each_line.split(' ')
                xml_files.write('   <object>\n')
                xml_files.write(f'      <name>{labels[int(line[0])]}</name>\n')
                xml_files.write('      <pose>Unspecified</pose>\n')
                xml_files.write('      <truncated>0</truncated>\n')
                xml_files.write('      <difficult>0</difficult>\n')
                xml_files.write('      <bndbox>\n')
                center_x = round(float(line[1]) * img_w)
                center_y = round(float(line[2]) * img_h)
                bbox_w = round(float(line[3]) * img_w)
                bbox_h = round(float(line[4]) * img_h)
                xmin = str(int(center_x - bbox_w / 2))
                ymin = str(int(center_y - bbox_h / 2))
                xmax = str(int(center_x + bbox_w / 2))
                ymax = str(int(center_y + bbox_h / 2))
                xml_files.write(f'         <xmin>{xmin}</xmin>\n')
                xml_files.write(f'         <ymin>{ymin}</ymin>\n')
                xml_files.write(f'         <xmax>{xmax}</xmax>\n')
                xml_files.write(f'         <ymax>{ymax}</ymax>\n')
                xml_files.write('      </bndbox>\n')
                xml_files.write('   </object>\n')
        xml_files.write('</annotation>')

