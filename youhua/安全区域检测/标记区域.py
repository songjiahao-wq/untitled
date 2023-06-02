import cv2
import imutils
import numpy as np
import shutil

# 定义全局变量
polygon1 = []
polygon2 = []
current_polygon = 1

# 鼠标事件回调函数
def mouse_event(event, x, y, flags, param):
    global polygon1, polygon2, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_polygon == 1:
            polygon1.append((x, y))
        elif current_polygon == 2:
            polygon2.append((x, y))

# 主函数
def main():
    global polygon1, polygon2, current_polygon

    # 读取视频
    cap = cv2.VideoCapture('D:\my_job\DATA\data/test.mp4')

    cv2.namedWindow('Video Frame')
    cv2.setMouseCallback('Video Frame', mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 在帧上绘制多边形
        for i, poly in enumerate([polygon1, polygon2]):
            if len(poly) > 1:
                cv2.polylines(frame, [np.array(poly)], isClosed=False, color=(0, 255, 0), thickness=2)
        # frame = imutils.resize(frame,width=980)
        # 显示帧
        cv2.imshow('Video Frame', frame)

        # 处理键盘输入
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_polygon += 1
            if current_polygon > 2:
                break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 打印多边形坐标
    print("Polygon 1:", polygon1)
    print("Polygon 2:", polygon2)


if __name__ == "__main__":
    main()
