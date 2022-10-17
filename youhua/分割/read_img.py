import cv2 as cv
import numpy as np
def image_normalization(img, img_min=0, img_max=255):
    """数据正则化,将数据从一个小范围变换到另一个范围
        默认参数：从（0,1） -> (0,255)

    :param img: 输入数据
    :param img_min: 数据最小值
    :param img_max: 数据最大值
    :return: 返回变换后的结果结果
    """
    img = np.float32(img)
    epsilon = 1e-12
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min

    return img


def show_img3(path):
    """ 利用opencv 读取并显示单通道图片

    :param path: 图片路径
    :return:
    """
    # 读取图片
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    # 将图片的值从一个小范围 转换到大范围
    img = image_normalization(img)
    # 改为uint8型
    img = img.astype('uint8')
    # 显示
    cv.imshow('single channel', img)
    cv.waitKey(0)

y_path = './fenge1.png'
show_img3(y_path)
