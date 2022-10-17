from pathlib import Path
import cv2
import os
#截图图像
def cutVideo(video_path):
    i = 0
    name_id = 0
    video = cv2.VideoCapture(video_path) # 读取视频文件
    cut_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / 6)
    while(True):
        ret,frame = video.read()
        if not ret:
            break
        cv2.imshow('video',frame)
        cv2.waitKey(1)
        i=i+1
        # 保存name
        name = Path(video_path).name
        if i% cut_frame==0:
            name_id += 1
            name = str(name) + '.' + str(name_id)
            cv2.imwrite(os.path.join(Path(video_path).parent,str(name)+'.jpg'),frame)

cutVideo(video_path=r'D:\yanyi\shixi\sangang\test\2.mp4')
cv2.destroyAllWindows()
