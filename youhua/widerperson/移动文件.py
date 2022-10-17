import shutil
import os
if __name__ == '__main__':
    label_path=r"D:\songjiahao\DATA\WiderPerson\widerperson\labels\train/"
    imgids = os.listdir(label_path)
    print(len(imgids))
    n=0
    for i in imgids:
        n += 1
        img_ids_path = label_path.replace('labels', 'images')+ i[0:6] +'.jpg'
        To_imgpath=r'D:\songjiahao\DATA\WiderPerson\widerperson\trainimg/'
        print(img_ids_path,To_imgpath,n)
        shutil.copy(img_ids_path, To_imgpath)
