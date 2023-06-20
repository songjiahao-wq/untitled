import os


def count_categories(directory):
    categories = {}

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                # 遍历文件的每一行
                for line in file:
                    # 提取每一行的第一个字符作为数据类别
                    category = line[0]

                    # 如果类别已存在于字典中，增加计数；如果不存在，添加到字典中并设置计数为1
                    if category in categories:
                        categories[category] += 1
                    else:
                        categories[category] = 1

    return categories


directory = r"D:\xian_yu\xianyun_lunwen\xiaci2000\datasets\labels\train"  # 修改为你的目录路径
categories = count_categories(directory)

for category, count in categories.items():
    print(f"Category: {category}, Count: {count}")
