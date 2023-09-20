import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# 定义文件夹路径
folder_path = r'F:\yolov5-master-Insulator\runs\insulator\tu'  # 替换为你的文件夹路径

# 获取文件夹内所有的Excel文件
excel_files = [os.path.join(folder_path,f,'results.csv') for f in os.listdir(folder_path)]

# 定义要读取的列
columns = ["   metrics/precision", "      metrics/recall", "     metrics/mAP_0.5",
           "metrics/mAP_0.5:0.95"]  # 这里替换为你的实际列名

# 为每个列索引生成一个折线图
for col in columns:
    plt.figure(figsize=(3, 3))

    # 对于每个文件，绘制该列的数据
    for file in excel_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        plt.plot(df[col], label=Path(file).parent.stem, linewidth=1.5)  # 调整线条粗细

    plt.title(col, fontsize=12, fontname='Times New Roman')  # 设置标题字体
    plt.xlabel('epoch', fontsize=10, fontname='Times New Roman')  # 设置x轴标签字体
    # plt.ylabel('value', fontsize=10, fontname='Times New Roman')  # 设置y轴标签字体
    legend = plt.legend(loc='lower right', prop={'size': 5})
    plt.setp(legend.get_texts(), fontname="Times New Roman")  # 设置图例字体
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=8, labelrotation=45)  # 调整刻度标签字体大小
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Times New Roman")
    save_path = os.path.join('./', f"{col.replace(' ', '').replace('/', '_').replace(':', '-')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前的绘图窗口
plt.show()



