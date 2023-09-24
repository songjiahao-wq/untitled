# -*- coding=utf-8 -*-

import math
import random
# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像(需要改变bbox)
#     7. cutout
# 注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!
# need_aug_num #扩充数量
# specific_name #增强特征类
import sys
import cv2
import numpy as np
from skimage import exposure
from skimage.util import random_noise
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image


# ros_path = './demo/demo1'
#
# if ros_path in sys.path:
#     sys.path.remove(ros_path)


# 显示带标签显示的图片
def show_pic(img, bboxes=None, labels=None):
    """
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    """
    img = img / 255
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
        cv2.putText(img, labels[i], (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # cv2.namedWindow('pic', 0)  # 1表示原图
    # cv2.moveWindow('pic', 0, 0)
    # cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    # cv2.imshow('pic', img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()

'''
************************************args:************************************
rotation_rate=0.5(旋转), max_rotation_angle=30（旋转）, crop_rate=0.5（裁剪）, shift_rate=0.5（平移）, change_light_rate=0.5（调整亮度）,change_saturation_rate=0.5（改变饱和度）,
add_noise_rate=0.5（加噪声）, flip_rate=0.5（翻转）,
cutout_rate=0.5（cutout）, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5
need_aug_num = 2 #扩充数量
specific_name = '' #增强特征类
'''
# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=30,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,change_saturation_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.change_saturation_rate = change_saturation_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    # 加噪声
    def _addNoise(self, img):
        """
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        # random.seed(int(time.time()))
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True) * 255

    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)

    def _changesaturation(self, img, saturation_factor):
        """

        :param img:
        :param saturation_factor: 饱和度调节,大于1为增强饱和度
        :return:
        """
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #缩放所有像素点的S通道
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255)
        #将图片转化会BGR
        output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return output_img

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):

        def cal_iou(boxA, boxB):
            """
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            """

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape

        mask = np.ones((h, w, c), np.float32)

        for n in range(n_holes):

            chongdie = True  # 看切割的区域是否与box重叠太多

            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)
                # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        """
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        """
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        """
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        """
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        # ---------------------- 裁剪后的图像大小 ----------------------
        crop_h, crop_w = crop_img.shape[0], crop_img.shape[1]

        # 计算每侧的填充量
        top_pad = crop_y_min
        bottom_pad = h - crop_y_max
        left_pad = crop_x_min
        right_pad = w - crop_x_max

        # 使用cv2.copyMakeBorder方法填充
        # 这里填充的颜色是黑色，可以按需要更改
        padded_img = cv2.copyMakeBorder(crop_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))

        # 如果你的bounding boxes是相对于填充后的图像的，则需要将bounding box的坐标调整到填充后的图像上
        for i in range(len(crop_bboxes)):
            crop_bboxes[i][0] += left_pad
            crop_bboxes[i][2] += left_pad
            crop_bboxes[i][1] += top_pad
            crop_bboxes[i][3] += top_pad
        return padded_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        """
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        """
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])
        # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        try:
            shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        except Exception as e:
            print("error")

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        """
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        """
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        #         if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
        horizon = True
        #         else:
        #             horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        """
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        """
        change_num = 0  # 改变的次数
        print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效
            if random.random() < self.crop_rate:  # 裁剪
                print('裁剪')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)

            if random.random() > self.rotation_rate:  # 旋转
                print('旋转')
                change_num += 1
                angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                #                 angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if random.random() < self.shift_rate:  # 平移
                print('平移')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if random.random() > self.change_light_rate:  # 改变亮度
                print('亮度')
                change_num += 1
                img = self._changeLight(img)

            if random.random() > self.change_saturation_rate:  # 改变饱和度
                print('饱和度')
                change_num += 1
                img = self._changesaturation(img, 1.5)

            if random.random() < self.add_noise_rate:  # 加噪声
                print('加噪声')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  # cutout
                print('cutout')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                   threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:  # 翻转
                print('翻转')
                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)

            print('\n')
        # print('------')
        return img, bboxes


# -*- coding=utf-8 -*-
import xml.etree.ElementTree as ET


# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    """
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(float(box[0].text))
        y_min = int(float(box[1].text))
        x_max = int(float(box[2].text))
        y_max = int(float(box[3].text))
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


# 保存xml文件函数的核心实现，输入为图片名称image_name,分类category（一个列表，元素与bbox对应），bbox(一个列表，与分类对应)，保存路径save_dir ，通道数channel
def save_xml(image_name, category, bbox, file_dir, save_dir, channel=3):
    file_path = file_dir
    img = Image.open(file_path + image_name)
    width = img.size[0]
    height = img.size[1]

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for i in range(len(bbox)):
        left, top, right, bottom = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = category[i]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


need_aug_num = 2 #扩充数量
specific_name = '' #增强特征类
dataAug = DataAugmentForObjectDetection()

source_pic_root_path = r'D:\my_job\code\xianyu\5.7\VOCdevkit3\VOC2007\JPEGImages/'  # 原图所在文件夹路径
source_xml_root_path = r'D:\my_job\code\xianyu\5.7\VOCdevkit3\VOC2007\Annotations/'  # 原XML文件所在文件夹路径
img_save_path = r'D:\my_job\code\xianyu\5.7\VOCdevkit3\VOC2007\augimages2/'  # 新图片存储路径
save_dir = r'D:\my_job\code\xianyu\5.7\VOCdevkit3\VOC2007\augAnnotions2/'  # 新xml存储路径
# 如果保存文件夹不存在就创建
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for parent, _, files in os.walk(source_pic_root_path):
    for file in files:
        cnt = 0
        print(file)
        while cnt < need_aug_num:
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
            if not os.path.exists(xml_path):
                break
            coords = parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            if specific_name =='':
                coordss = [coord[:4] for coord in coords]
            else:
                print("增强特定类")
                coordss = [coord[:4] for coord in coords if coord[4] == specific_name]
            labels = [coord[4] for coord in coords]
            img = cv2.imread(pic_path)
            show_pic(img, coordss, labels)  # 原图

            auged_img, auged_bboxes = dataAug.dataAugment(img, coordss)
            cnt += 1

            cv2.imwrite(img_save_path + file[:-4]+'_'+ str(cnt)+ '.jpg', auged_img)
            save_xml(file[:-4]+'_' +str(cnt) + '.jpg', labels, auged_bboxes, file_dir=img_save_path, save_dir=save_dir)
            show_pic(auged_img, auged_bboxes, labels)  # 强化后的图
cv2.destroyAllWindows()