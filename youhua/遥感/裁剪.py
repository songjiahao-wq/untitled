import sys
import os
import cv2
import numpy as np
import os.path
from xml.dom.minidom import Document
from tqdm import tqdm
import xml.etree.cElementTree as ET


def clip_img(number, oriname):
    # ------------------------------------------------------------#
    # 读入原始图像
    # 保存原图的大小
    # 可以resize也可以不resize，看情况而定
    # ------------------------------------------------------------#
    from_name = os.path.join(image_path, oriname + '.jpg')
    img = cv2.imread(from_name)
    h_ori, w_ori, _ = img.shape
    img = cv2.resize(img, (2048, 1536))
    h, w, _ = img.shape

    # ------------------------------------------------------------#
    # 输入.xml文件
    # 创建存放坐标四个值和类别的列表
    # ------------------------------------------------------------#
    xml_name = os.path.join(label_xml_path, oriname + '.xml')
    xml_ori = ET.parse(xml_name).getroot()
    res = np.empty((0, 5))

    # ------------------------------------------------------------#
    # 找到每个.xml文件中的bbox
    # lower().strip()转化小写移除字符串头尾空格
    # vstack() 水平堆叠
    # ------------------------------------------------------------#
    for obj in xml_ori.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            cur_pt = int(cur_pt * h / h_ori) if i % 2 == 1 else int(cur_pt * w / w_ori)
            bndbox.append(cur_pt)

        bndbox.append(name)
        res = np.vstack((res, bndbox))
    print('*' * 5, res)

    # -------------------------------------------------------------#
    # 开始剪切 + 写入标签信息
    # -------------------------------------------------------------#
    i = 0
    win_size = 512  # 分块的大小
    stride = 128  # 重叠的大小，设置这个可以使分块有重叠
    for r in range(0, h - win_size, stride):
        for c in range(0, w - win_size, stride):
            flag = np.zeros([1, len(res)])

            youwu = False
            xiefou = True

            tmp = img[r: r + win_size, c: c + win_size]
            for re in range(res.shape[0]):
                xmin, ymin, xmax, ymax, label = res[re]
                # ------------------------------------------------#
                # 判断bb是否在当前剪切的区域内
                # ------------------------------------------------#
                if int(xmin) >= c and int(xmax) <= c + win_size and int(ymin) >= r and int(ymax) <= r + win_size:
                    flag[0][re] = 1
                    youwu = True
                elif int(xmin) < c or int(xmax) > c + win_size or int(ymin) < r or int(ymax) > r + win_size:
                    pass
                else:
                    xiefou = False
                    break

            # 如果物体被分割了，则忽略不写入
            if xiefou:
                # 有物体则写入xml文件
                if youwu:
                    # ---------------------------------------------------#
                    # 创建.xml文件 + 写入bb
                    # ---------------------------------------------------#
                    doc = Document()

                    width, height, channel = str(win_size), str(win_size), str(3)

                    annotation = doc.createElement('annotation')
                    doc.appendChild(annotation)

                    size_chartu = doc.createElement('size')
                    annotation.appendChild(size_chartu)

                    width1 = doc.createElement('width')
                    width1_text = doc.createTextNode(width)
                    width1.appendChild(width1_text)
                    size_chartu.appendChild(width1)

                    height1 = doc.createElement('height')
                    height1_text = doc.createTextNode(height)
                    height1.appendChild(height1_text)
                    size_chartu.appendChild(height1)

                    channel1 = doc.createElement('channel')
                    channel1_text = doc.createTextNode(channel)
                    channel1.appendChild(channel1_text)
                    size_chartu.appendChild(channel1)

                    for re in range(res.shape[0]):

                        xmin, ymin, xmax, ymax, label = res[re]

                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)

                        if flag[0][re] == 1:

                            xmin = str(xmin - c)
                            ymin = str(ymin - r)
                            xmax = str(xmax - c)
                            ymax = str(ymax - r)

                            object_charu = doc.createElement('object')
                            annotation.appendChild(object_charu)

                            name_charu = doc.createElement('name')
                            name_charu_text = doc.createTextNode(label)
                            name_charu.appendChild(name_charu_text)
                            object_charu.appendChild(name_charu)

                            dif = doc.createElement('difficult')
                            dif_text = doc.createTextNode('0')
                            dif.appendChild(dif_text)
                            object_charu.appendChild(dif)

                            bndbox = doc.createElement('bndbox')
                            object_charu.appendChild(bndbox)

                            xmin1 = doc.createElement('xmin')
                            xmin_text = doc.createTextNode(xmin)
                            xmin1.appendChild(xmin_text)
                            bndbox.appendChild(xmin1)

                            ymin1 = doc.createElement('ymin')
                            ymin_text = doc.createTextNode(ymin)
                            ymin1.appendChild(ymin_text)
                            bndbox.appendChild(ymin1)

                            xmax1 = doc.createElement('xmax')
                            xmax_text = doc.createTextNode(xmax)
                            xmax1.appendChild(xmax_text)
                            bndbox.appendChild(xmax1)

                            ymax1 = doc.createElement('ymax')
                            ymax_text = doc.createTextNode(ymax)
                            ymax1.appendChild(ymax_text)
                            bndbox.appendChild(ymax1)

                        else:
                            continue
                    xml_name = oriname + '_%3d.xml' % (i)
                    to_xml_name = os.path.join(lablel_xml_path, xml_name)
                    with open(to_xml_name, 'wb+') as f:
                        f.write(doc.toprettyxml(indent="\t", encoding='utf-8'))
                    # name = '%02d_%02d_%02d_.bmp' % (number, int(r/win_size), int(c/win_size))
                    img_name = oriname + '_%3d.png' % (i)
                    to_name = os.path.join(image_crop_path, img_name)
                    i = i + 1
                    cv2.imwrite(to_name, tmp)


if __name__ == "__main__":

    image_path = r'D:\songjiahao\DATA\crowhuman-voc\VOC2007\JPEGImages'
    label_xml_path = r'D:\songjiahao\DATA\crowhuman-voc\VOC2007\Annotations'

    image_crop_path = r'D:\songjiahao\DATA\crowhuman-voc\test\\image\\'
    lablel_xml_path = r'D:\songjiahao\DATA\crowhuman-voc\test\\label\\'

    if not os.path.exists(image_crop_path):
        os.makedirs(image_crop_path)
    if not os.path.exists(lablel_xml_path):
        os.makedirs(lablel_xml_path)

    for i, name in tqdm(enumerate(os.listdir(image_path))):
        clip_img(i, name.rstrip('.jpg'))


