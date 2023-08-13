import  os
import random

'''
将图片划分为训练集和测试集
保存形式为train.txt val.txt
'''
def writ_save(train_val_list:list,train_or_val:str):
    with open(train_or_val,"w") as f:
        for i in train_val_list:
            i_ = i + "\n"
            f.write(i_)

def split_data(image_path:str,split_rate:int):
    '''
    划分数据集
    :param image_path: 图片文件夹路径
    :param split_rate: 划分为训练集的比例（0-1）
    :return:
    '''
    new_image_name_list = []
    for image_name in os.listdir(image_path):
        new_image_name = image_name.split(".")[0]
        new_image_name_list.append(new_image_name)
    # 按比例划分为训练集
    train_num = int(len(new_image_name_list) * split_rate)
    train_list = random.sample(new_image_name_list,train_num)

    #剩下的作为测试集
    val_list = []
    for new_image_name in new_image_name_list:
        if new_image_name not in train_list:
            val_list.append(new_image_name)

    return train_list,val_list



if __name__ == "__main__":
    train_list,val_list = split_data(r"D:\xian_yu\xianyun_lunwen\x guang jian ce\xf_Xray\train\img",0.85)
    writ_save(train_list, r"D:\xian_yu\xianyun_lunwen\x guang jian ce\xf_Xray\train.txt")
    writ_save(val_list, r"D:\xian_yu\xianyun_lunwen\x guang jian ce\xf_Xray\val.txt")
