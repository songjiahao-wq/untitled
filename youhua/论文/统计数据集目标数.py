# 1、统计数据集中小、中、大 GT的个数
# 2、统计某个类别小、中、大 GT的个数
# 3、统计数据集中ss、sm、sl GT的个数
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体为微软雅黑
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimSun'

def getGtAreaAndRatio(label_dir):
    """
    得到不同尺度的gt框个数
    :params label_dir: label文件地址
    :return data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    """
    data_dict = {}
    assert Path(label_dir).is_dir(), "label_dir is not exist"

    txts = os.listdir(label_dir)  # 得到label_dir目录下的所有txt GT文件

    for txt in txts:  # 遍历每一个txt文件
        with open(os.path.join(label_dir, txt), 'r') as f:  # 打开当前txt文件 并读取所有行的数据
            lines = f.readlines()

        for line in lines:  # 遍历当前txt文件中每一行的数据
            temp = line.split()  # str to list{5}
            coor_list = list(map(lambda x: x, temp[1:]))  # [x, y, w, h]
            area = float(coor_list[2]) * float(coor_list[3])  # 计算出当前txt文件中每一个gt的面积
            # center = (int(coor_list[0] + 0.5*coor_list[2]),
            #           int(coor_list[1] + 0.5*coor_list[3]))
            ratio = round(float(coor_list[2]) / float(coor_list[3]), 2)  # 计算出当前txt文件中每一个gt的 w/h

            if temp[0] not in data_dict:
                data_dict[temp[0]] = {}
                data_dict[temp[0]]['area'] = []
                data_dict[temp[0]]['ratio'] = []

            data_dict[temp[0]]['area'].append(area)
            data_dict[temp[0]]['ratio'].append(ratio)

    return data_dict

def getSMLGtNumByClass(data_dict, class_num):
    """
    计算某个类别的小物体、中物体、大物体的个数
    params data_dict: {dict: 3}  3 x {'类别':{’area':[...]}, {'ratio':[...]}}
    params class_num: 类别  0, 1, 2
    return s: 该类别小物体的个数  0 < area <= 32*32
           m: 该类别中物体的个数  32*32 < area <= 96*96
           l: 该类别大物体的个数  area > 96*96
    """
    s, m, l = 0, 0, 0
    # print(data_dict)
    for item in data_dict['{}'.format(class_num)]['area']:
        if item * 640 * 640 <= 32 * 32:
            s += 1
        elif item * 640 * 640 <= 96 * 96:
            m += 1
        else:
            l += 1
    return s, m, l

def getAllSMLGtNum(data_dict):
    """
    数据集所有类别小、中、大GT分布情况
    """
    S, M, L = 0, 0, 0
    for i in range(1):
        s, m, l = getSMLGtNumByClass(data_dict, i)
        S += s
        M += m
        L += l
    return [S, M, L]

def analyAllSmallGt(data_dict):
    ss, sm, sl = 0, 0, 0
    for c in range(3):
        for item in data_dict['{}'.format(c)]['area']:
            if item * 640 * 640 <= 8 * 8:
                ss += 1
            elif item * 640 * 640 <= 16 * 16:
                sm += 1
            elif item * 640 * 640 <= 32 * 32:
                sl += 1
    return [ss, sm, sl]


# 画图函数
def plotAllSML(SML):
    x = ['S:[0, 32x32]', 'M:[32x32, 96x96]', 'L:[96×96, 640x640]']
    fig = plt.figure(figsize=(10, 8))  # 画布大小和像素密度
    plt.bar(x, SML, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, SML, range(len(x))):  # zip 函数
        plt.text(a, b + 0.01, "%d" % int(SML[i]), ha='center', fontsize=15, color="r")  # plt.text 函数
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('gt大小', fontsize=16)
    plt.ylabel('数量', fontsize=16)
    plt.title('广佛手病虫害训练集小、中、大GT分布情况(640x640)', fontsize=16)
    plt.show()
    # 保存到本地
    # plt.savefig("")

def plotSMLByClass(sml, c):
    if c == 0:
        txt = 'person'
    elif c == 1:
        txt = 'Medium'
    elif c == 2:
        txt = 'Large'

    x = ['S:[0, 32x32]', 'M:[32x32, 96x96]', 'L:[96*96, 640x640]']
    plt.figure(figsize=(6, 4),dpi=80)  # 画布大小和像素密度
    plt.bar(x, sml, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, sml, range(len(x))):  # zip 函数
        plt.text(a, b + 0.01, "%d" % int(sml[i]), ha='center', fontsize=15, color="r")  # plt.text 函数
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('gt大小', fontsize=20)
    plt.ylabel('数量', fontsize=20)
    plt.title('{}:小、中、大GT分布情况(640x640)'.format(txt), fontsize=20)
    plt.show()
    # 保存到本地
    # plt.savefig("")

def plotAllSmallGt(sml):
    x = ['ss:[0, 8x8]', 'sm:[8x8, 16x16]', 'sl:[16x16, 32x32]']
    fig = plt.figure(figsize=(10, 8))  # 画布大小和像素密度
    plt.bar(x, sml, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, sml, range(len(x))):  # zip 函数
        plt.text(a, b + 0.01, "%d" % int(sml[i]), ha='center', fontsize=15, color="r")  # plt.text 函数
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('gt大小', fontsize=16)
    plt.ylabel('数量', fontsize=16)
    plt.title('广佛手ss、sm、sl GT分布情况(640x640)', fontsize=16)
    plt.show()
    # 保存到本地
    # plt.savefig("")


if __name__ == '__main__':
    labeldir = r'F:\sjh\DATA\wider+crowd\labels\train'
    data_dict = getGtAreaAndRatio(labeldir)
    # 1、数据集所有类别小、中、大GT分布情况
    # SML = getAllSMLGtNum(data_dict)
    # print(SML)
    # plotAllSML(SML)

    # 2、数据集某个类别小中大GT分布情况
    # 0: 白粉病 powdery_mildew
    # 1: 潜叶蛾 leaf_miner
    # 2: 炭疽病 anthracnose
    c = 0
    sml = getSMLGtNumByClass(data_dict, c)
    plotSMLByClass(sml, c)
    # a(data_dict)

    # 3、分析所有小目标样本的一个分布
    # ss: 0<area<8x8  sm: 8x8<area<16x16  sl: 16x16<area<32x32
    # p = analyAllSmallGt(data_dict)
    # plotAllSmallGt(p)
