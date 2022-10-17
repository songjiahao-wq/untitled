import cv2
import os
#要提取视频的文件名，隐藏后缀
root_path="D:\yanyi\shixi\sangang/"#地址
sourceFileName='14-48-54' #文件名
#在这里把后缀接上
video_path = os.path.join(root_path, sourceFileName+'.mp4')
times=0
#提取视频的频率，每１帧提取一个
frameFrequency=3
# 起始ID文件名
start_ID=7405 #
#输出图片到当前目录vedio文件夹下
outPutDirName=root_path+sourceFileName+'/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(video_path)
while True:
    times+=1
    print('总数：',times/frameFrequency)
    res, image = camera.read()
    if not res:
        print('not res , not image')
        print(video_path)
        break
    if times%frameFrequency==0:
        cv2.imwrite(outPutDirName + str(start_ID)+'.jpg', image)
        # print(outPutDirName + str(times)+'.jpg')
        print(outPutDirName + str(start_ID)+'.jpg')
        start_ID+=1
print('图片提取结束')
camera.release()
