# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os, cv2

xml_file = r'F:\sjh\DATA2\xianyu\smoke_Phone\1smoke\train\xml/000-0.xml'
tree = ET.parse(xml_file)
root = tree.getroot()
imgfile = r'F:\sjh\DATA2\xianyu\smoke_Phone\1smoke\train\images/000-0.jpg'
im = cv2.imread(imgfile)
for object in root.findall('object'):
    object_name = object.find('name').text
    Xmin = int(object.find('bndbox').find('xmin').text)
    Ymin = int(object.find('bndbox').find('ymin').text)
    Xmax = int(object.find('bndbox').find('xmax').text)
    Ymax = int(object.find('bndbox').find('ymax').text)
    color = (4, 250, 7)
    cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (6, 230, 230), 2)
    cv2.imshow('01', im)
    cv2.waitKey(0)
cv2.imwrite('。/02.jpg', im)