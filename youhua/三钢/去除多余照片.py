import os
imgpath=r'D:\app\PotPlayer\Capture\Times7.5_1\MyVideo_2'
labelpath=r'D:\app\PotPlayer\Capture\Times7.5_1\MyVideo_2txt'

imglist=os.listdir(imgpath)
labellist=os.listdir(labelpath)
for i,name in enumerate(imglist):
    imglist[i] = name.split('.')[0]
for j,name in enumerate(labellist):
    labellist[j] = name.split('.')[0]
imglist = sorted([int(i) for i in imglist])
labellist = sorted([int(i) for i in labellist])

id = 1
print(imglist)
print(labellist)
for i in imglist:
    if i in labellist:
        print(id)
        id +=1
    else:
        os.remove(imgpath + '/'+ str(i) + '.jpg')
        pass