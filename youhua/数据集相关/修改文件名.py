import os
ori_path = r'D:\songjiahao\DATA\Phone\phonecall_images'

dir_list = os.listdir(ori_path)
for i in dir_list:
    old_name = ori_path+os.sep + i
    new_name = ori_path + os.sep + i.replace('.png','.jpg')
    os.rename(old_name,new_name)
    print(old_name,'======>',new_name)