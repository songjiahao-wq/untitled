import cv2
import time
from pathlib import Path
"""本代码主要实现读取长视频，按照一定的帧数分成小的视频保存。只需修改代码中的视频路径，存储路径和想要保存的帧数即可"""
# 获取时间
def get_time(stream):
    it = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 不带分隔符的时间 可以用作文件名
    ft = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 时间格式
    name = Path(stream).name.split('.')[0] #视频名递增格式
    return [it, ft,name]
def cutVideo(stream,video_path):

    cap = cv2.VideoCapture(stream)  # 读入视频
    # 获取高 宽 帧率
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = min(cap.get(cv2.CAP_PROP_FPS) % 100, 2) or 25.0

    if cap.isOpened() == False or w == 0 or h == 0:
        print("connect failed")

    frame_count = 0  # 记录帧数
    print(get_time(stream)[2])
    temp_path = video_path + get_time(stream)[2] + "_2.mp4" # 保存路径 （拼上时间作为文件名）
    cut_frame =int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 10)
    vid_writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
        cap.grab()  # 获取下一帧
        success, im = cap.retrieve()  # 解码
        if success:
            frame_count += 1  # 帧数加1
        else:
            vid_writer.release()  # 保存到本地
            break

        if (frame_count % cut_frame) == 0:  # 设定一个时间 多久保存一次
            vid_writer.write(im)
            cv2.waitKey(1)
            print(frame_count)
            continue

    cap.release()
    print("ok")


stream = r'D:\yanyi\shixi\sangang\test\2.mp4'

video_path = r'D:\yanyi\shixi\sangang\test/'  # 视频保存地址
cutVideo(stream,video_path)