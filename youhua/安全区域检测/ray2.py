import matplotlib.pyplot as plt



# 绘制多

def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for j, data in enumerate(poly):
        for i ,corner in enumerate(data):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:  # if point is on edge
                    is_in = True
                    break
                elif x > px:  # if point is on left-side of line
                    is_in = not is_in
    return is_in


if __name__ == '__main__':
    point = [0.5, 0.5]
    poly = [[0, 0], [0, 5], [5, 5], [5, 0]],[[0, 0],[1,1],[1, 3],  [3,3], [4,1],[1,1]]
    print(is_in_poly(point, poly))

    # 绘制多边形
    poly.append(poly[0])  # 添加第一个点，使多边形闭合
    poly_x, poly_y = zip(*poly)  # 解压多边形的坐标
    plt.plot(poly_x, poly_y, marker='o', linestyle='-', markersize=5, label='Polygon')

    # 绘制点
    plt.plot(point[0], point[1], marker='o', color='r', markersize=5, label='Point (3, 3)')

    # 设置绘图区域的边界和比例
    plt.xlim(-1, 9)
    plt.ylim(-1, 9)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend()
    plt.show()

