#
import os
import shutil

img_path = r'D:\app\PotPlayer\Capture\Times7.5_1\MyVideo_2'
anno_path = r'D:\app\PotPlayer\Capture\Times7.5_1\MyVideo_2txt'
new_path = r'D:\app\PotPlayer\Capture\Times7.5_1\images'
star_id=1103
if not os.path.exists(new_path):
    os.mkdir(new_path)
name_list=os.listdir(anno_path)
for name in name_list:
    name=name.split('.')[0]
    shutil.move(img_path+'/'+name+'.jpg',new_path+'/'+name+'.jpg')
    print(img_path+'/'+name+'.jpg',new_path+'/'+name+'.jpg')
