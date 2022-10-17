import os
import xml.etree.ElementTree as ET
import shutil
import glob
import cv2
dirpath = r'/home/data/831/'  # 原来存放xml文件的目录
newdir = r'/home/data/832/labels/train'  # 修改label后形成的txt目录
classes = [ 'hat', 'person']




def moveimg():
    total_img = glob.glob(os.path.join(dirpath,'*.jpg'))
    for fp in total_img:
        shutil.copy(fp,'/home/data/832/images/train/')
def mkdir():
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    os.makedirs('/home/data/832/images/train')
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    if w>=1:
        w=0.99
    if h>=1:
        h=0.99
    return (x,y,w,h)

def convert_annotation(rootpath,xmlname):
    xmlfile = os.path.join(xmlpath,xmlname)
    with open(xmlfile, "r", encoding='UTF-8') as in_file:
        txtname = xmlname[:-4]+'.txt'
        print(txtname)
        txtpath = r'/home/data/832/labels/train'#生成的.txt文件会被保存在这里
        if not os.path.exists(txtpath):
            os.makedirs(txtpath)
        txtfile = os.path.join(txtpath,txtname)
        with open(txtfile, "w+" ,encoding='UTF-8') as out_file:
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            out_file.truncate()
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult)==1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ =='__main__':
    mkdir()
    moveimg()
    xmlpath = r'/home/data/831/'  # 原来存放xml文件的目录
    list = os.listdir(xmlpath)
    for i in range(0, len(list)):
        path = os.path.join(xmlpath, list[i])
        if ('.xml' in path) or ('.XML' in path):
            convert_annotation(xmlpath, list[i])
            print('done', i)
        else:
            print('not xml file', i)
