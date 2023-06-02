points = [(1086, 247), (625, 568), (976, 781), (1431, 501), (1085, 249)]

# 将所有点展平为一个列表
flattened_points = [list(point) for point in points ]
print(flattened_points)
# 使用逗号连接展平的点列表
formatted_points = ', '.join(flattened_points)

print(formatted_points)