import cv2
import numpy as np

# 定义全局变量
polygon = []
current_frame = None

# 鼠标事件回调函数
def mouse_event(event, x, y, flags, param):
    global polygon, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon.append((x, y))
        frame_with_polygon = current_frame.copy()

        if len(polygon) > 1:
            cv2.polylines(frame_with_polygon, [np.array(polygon)], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow('Video Frame', frame_with_polygon)

# 主函数
def main():
    global polygon, current_frame

    # 读取视频
    cap = cv2.VideoCapture('D:\my_job\DATA\data/test.mp4')

    cv2.namedWindow('Video Frame')
    cv2.setMouseCallback('Video Frame', mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()

        # 在帧上绘制多边形
        if len(polygon) > 1:
            cv2.polylines(frame, [np.array(polygon)], isClosed=False, color=(0, 255, 0), thickness=2)

        # 显示帧
        cv2.imshow('Video Frame', frame)

        # 处理键盘输入
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 打印多边形坐标
    print("Polygon:", polygon)

if __name__ == "__main__":
    main()
