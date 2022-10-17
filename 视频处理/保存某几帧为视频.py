import cv2
from pathlib import Path
# 将视频文件路径转化为标准的路径
def cutVideo(videoPath):
    times = 0
    videoPath=r'D:\yanyi\shixi\sangang/1.mp4'
    videoPath2=r'D:\yanyi\shixi\sangang/2.mp4'
    videoCapture=cv2.VideoCapture(videoPath)
    save_path = Path(videoPath)

    vid_writer = cv2.VideoWriter(videoPath2, cv2.VideoWriter_fourcc(*'mp4v'), 30, (480, 480))
    # 总帧数(frames)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    cutframe = int(frames/5)

    while True:
        times += 1
        res, image = videoCapture.read()
        if times % cutframe == 0:
            vid_writer.write(image)





p=1
cutVideo(p)