import cv2
# videoPath=input("请输入视屏文件的绝对路径：")
# 将视频文件路径转化为标准的路径
videoPath=r'D:\yanyi\shixi\sangang\test/3.mp4'
# 视屏获取
videoCapture=cv2.VideoCapture(videoPath)
# 帧率(frames per second)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
# 总帧数(frames)
frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print("帧数："+str(fps))
print("总帧数："+str(frames))
print("视屏总时长："+"{0:.2f}".format(frames/fps)+"秒")
