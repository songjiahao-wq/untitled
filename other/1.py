# import os
# trainval_path = r'D:\songjiahao\DATA\other\RoadDamageDataset\Ichihara\ImageSets\Main\trainval.txt'
# list_file = open(r'D:\songjiahao\DATA\other\RoadDamageDataset\Ichihara\train.txt', 'w')
# image_ids = open(trainval_path).read().strip().split()
# for image_id in image_ids:
#     list_file.write('D:/songjiahao/DATA/other/RoadDamageDataset/Ichihara/JPEGImages/%s.jpg\n' %(image_id))
#
# import os
# cls = []
# trainval_path = r'D:\songjiahao\DATA\other\RoadDamageDataset\Ichihara\ImageSets\Main\trainval.txt'
# image_ids = open(trainval_path).read().strip().split()
# for image_id in image_ids:
#     list_file=open('D:/songjiahao/DATA/other/RoadDamageDataset/Ichihara/labels/%s.txt' % (image_id),'r')
#     read_line = list_file.readline()
#     while read_line:
#         if read_line[0] not in cls:
#             cls.append(read_line[0])
#         read_line = list_file.readline()
# print(cls)

import os
cls = []
import xml.etree.ElementTree as ET
dirpath = r'D:\songjiahao\DATA\other\RoadDamageDataset\Ichihara\Annotations/'  # 原来存放xml文件的目录
trainval_path = r'D:\songjiahao\DATA\other\RoadDamageDataset\Ichihara\ImageSets\Main\trainval.txt'
image_ids = open(trainval_path).read().strip().split()
for image_id in image_ids:
    list_file = os.path.join('D:/songjiahao/DATA/other/RoadDamageDataset/Ichihara/Annotations/%s.xml' % (image_id))
    root = ET.parse(list_file).getroot()
    for child in root.findall('object'):
        name = child.find('name')

        if name.text not in cls:
            cls.append(name.text)
print(sorted(cls))