# 6.6
import shutil
import os
if __name__ == '__main__':
    label_path=r"D:\yanyi\shixi\puyang\new_zifu\labels/"
    imgids = os.listdir(label_path)
    print(len(imgids))
    n=0
    for i in imgids:
        n += 1
        name = i.split('.')[0]
        print(name)
        img_path = r'D:\yanyi\shixi\puyang\new_zifu\images/' + str(name) + '.jpg'
        print(img_path)
        to_path = r'D:\yanyi\shixi\puyang\new_zifu\now_images'
        if not os.path.exists(to_path):
            os.makedirs(to_path)
        # img_ids_path = label_path.replace('labels', 'images')+ i[0:6] +'.jpg'
        # To_imgpath=r'D:\songjiahao\DATA\WiderPerson\widerperson\trainimg/'
        # print(img_ids_path,To_imgpath,n)
        shutil.copy(img_path, to_path)
