import cv2
import numpy as np
#2023.5.1 这是一个测试好的生成两个安全区域的代码
# 定义全局变量
polygons = [[]]
current_frame = None
max_polygons = 2

# 鼠标事件回调函数
def mouse_event(event, x, y, flags, param):
    global polygons, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        polygons[-1].append((x, y))
        frame_with_polygons = current_frame.copy()

        for poly in polygons:
            if len(poly) > 1:
                cv2.polylines(frame_with_polygons, [np.array(poly)], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow('Video Frame', frame_with_polygons)

# 主函数
def main():
    global polygons, current_frame, max_polygons

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
        for poly in polygons:
            if len(poly) > 1:
                cv2.polylines(frame, [np.array(poly)], isClosed=False, color=(0, 255, 0), thickness=2)

        # 显示帧
        cv2.imshow('Video Frame', frame)

        # 处理键盘输入
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            if len(polygons) < max_polygons:
                polygons.append([])
        # 打印多边形坐标
        for i, polygon in enumerate(polygons, 1):
            print('***********************************************')
            print(f"Polygon {i}:", polygon)
            # 将所有点展平为一个列表
            flattened_points = [str(coordinate) for point in polygon for coordinate in point]
            flattened_points2 = [list(point) for point in polygon ]

            # 使用逗号连接展平的点列表
            formatted_points = ', '.join(flattened_points)
            print(f"Polygon {i}***str:", formatted_points)
            print(f"Polygon {i}***list:", flattened_points2)

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 打印多边形坐标
    for i, polygon in enumerate(polygons, 1):
        print(f"Polygon {i}:", polygon)

if __name__ == "__main__":
    main()
