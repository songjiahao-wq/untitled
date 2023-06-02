import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm

classes = ["TBbacillus"]  # 类别
xml_path = r"E:\tuberculosis-phonecamera\new\xml"
txt_path = r"E:\tuberculosis-phonecamera\new\labels"
image_path = r"E:\tuberculosis-phonecamera\new\images"


# 将原有的xmax,xmin,ymax,ymin换为x,y,w,h
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 输入时图像和图像的宽高
def convert_annotation(image_id, width, hight):
    in_file = open(xml_path + '\\{}.xml'.format(image_id), encoding='UTF-8')
    out_file = open(txt_path + '\\{}.txt'.format(image_id), 'w')  # 生成同名的txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')  # 此处是获取原图的宽高，便于后续的归一化操作
    if size is not None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    else:
        w = width
        h = hight

    for obj in root.iter('object'):
        cls = obj.find('label').text
        # print(cls)
        if cls not in classes:  # 此处会将cls里没有的类别打印，以便后续添加
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 此处获取图像宽高的数组，tqdm为进度条库，将处理可视化
def image_size(path):
    image = os.listdir(path)
    w_l, h_l = [], []
    for i in tqdm(image):
        if i.endswith('jpg'):
            h_l.append(cv2.imread(os.path.join(path, i)).shape[0])
            w_l.append(cv2.imread(os.path.join(path, i)).shape[1])
    return w_l, h_l


# 遍历xml文件，将对应的宽高输入convert_annotation方法
if __name__ == "__main__":
    img_xmls = os.listdir(xml_path)
    w, h = image_size(image_path)
    i = 0
    for img_xml in img_xmls:
        label_name = img_xml.split('.')[0]
        print(label_name)
        convert_annotation(label_name, w[i], h[i])
        i += 1
