import cv2
import time
"""本代码主要实现读取长视频，按照一定的帧数分成小的视频保存。只需修改代码中的视频路径，存储路径和想要保存的帧数即可"""

# 获取时间
def get_time():
    it = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 不带分隔符的时间 可以用作文件名
    ft = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 时间格式
    return [it, ft]


stream = 'k400datasets/fight1.mp4'

video_path = 'k400datasets/video1/'  # 视频保存地址
cap = cv2.VideoCapture(stream)  # 读入视频
# 获取高 宽 帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 25.0

if cap.isOpened() == False or w == 0 or h == 0:
    print("connect failed")
num = 300
frame_count = 0  # 记录帧数
# temp_path = video_path + get_time()[0] + ".mp4" # 保存路径 （拼上时间作为文件名）

while True:
    if (frame_count % num) == 0:
        temp_path = video_path + get_time()[0] + ".mp4"
        vid_writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.grab()  # 获取下一帧
    success, im = cap.retrieve()  # 解码
    if success:
        frame_count += 1  # 帧数加1
    vid_writer.write(im)
    cv2.imshow("camera", im)
    cv2.waitKey(1)
    if (frame_count % num) == 0:  # 设定一个时间 多久保存一次
        vid_writer.release()  # 保存到本地
        continue
cap.release()
print("ok")