#第一天学习
#第一天学习
#对文件名重新顺序命名
import os

imgpath = r'D:\my_job\code\xianyu\datasets\Pneumonia\images\train/'
labelpath = r'D:\my_job\code\xianyu\datasets\Pneumonia\labels\train/'

imglist = os.listdir(imgpath)
labellist = os.listdir(labelpath)
imgn = 00000
laben = 00000

# imglist.sort(key=lambda x: int(x[:-4])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型

# print(sorted(labellist))
for imaname in imglist:

    img_id = '%05d' %imgn
    imgname = imaname.strip('.jpg')
    oriimg= imgpath + imgname + '.jpg'
    newimname = str(img_id) + '.jpg'
    newimg = imgpath +newimname
    # print(oriimg,newimg)




    orilabel = labelpath + imgname + '.txt'
    newtxtname = str(img_id) + '.txt'
    newlabel = labelpath + newtxtname
    # print(orilabel,newlabel)
    imgn += 1
    if not os.path.exists(oriimg) or not os.path.exists(orilabel):
        print("no exists img",oriimg,orilabel)
        continue
    elif oriimg == newimg:
        print("Has been renamed：",oriimg,newimg)
        continue

    img_size = os.path.getsize(oriimg)
    label_size = os.path.getsize(orilabel)
    if img_size == 0 or label_size ==0:
        print('文件是空的')
        print(orilabel,'文件已删除')
        os.remove(orilabel)
        os.remove(oriimg)
        continue
    print(newimg,newlabel)
    os.rename(oriimg, newimg)
    os.rename(orilabel, newlabel)