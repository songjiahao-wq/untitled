import os
import random
# 实现了将xml格式的标签文件转换为txt文件，同时按比例划分为训练集和验证集的功能。
oriImgDir = "./change14img"
oriXmlDir = "./change14xml"    #源文件夹
desTestImgDir = "./insect/images/test"   #目标文件夹
desValImgDir = "./insect/images/val"
desTrainImgDir = "./insect/images/train"
desTestXmlDir = "./insect/labels/test"
desValXmlDir = "./insect/labels/val"
desTrainXmlDir = "./insect/labels/train"
# 获取目录下文件名清单
flist = os.listdir(oriImgDir)
random.shuffle(flist)
length = len(flist)
val = int(length * 0.2)
test = int(length * 0.2)

count = 0

while (count < length):
    # 移动测试集
    while test:
        item = flist[count]
        fname = os.path.splitext(item)[0]  # 获得不带后缀的文件名
        os.rename(oriImgDir + "/" + item, desTestImgDir+"/14"+str(count)+".jpg")  # 图片
        os.rename(oriXmlDir + "/" + fname + ".xml", desTestXmlDir+"/14"+str(count)+".xml")
        test -= 1
        count += 1
    # 移动验证集
    while val:
        item = flist[count]
        fname = os.path.splitext(item)[0]  # 获得不带后缀的文件名
        os.rename(oriImgDir + "/" + item, desValImgDir+"/14"+str(count)+".jpg")  # 图片
        os.rename(oriXmlDir + "/" + fname + ".xml", desValXmlDir+"/14"+str(count)+".xml")
        val -= 1
        count += 1
    # 剩下的移到训练集
    item = flist[count]
    fname = os.path.splitext(item)[0]
    os.rename(oriImgDir + "/" + item, desTrainImgDir+"/14"+str(count)+".jpg")  # 图片
    os.rename(oriXmlDir + "/" + fname + ".xml", desTrainXmlDir+"/14"+str(count)+".xml")
    count += 1

