import os## 标题
import shutil


def reName(oldPath, newPath):
    # 判断路径是否存在
    if os.path.exists(oldPath):
        # 获取该目录下所有文件，存入列表中
        fileList = os.listdir(oldPath)
        n = 0
        for i in fileList:

            # 设置旧文件名（就是路径+文件名）
            oldname = oldPath + os.sep + fileList[n]  # os.sep添加系统分隔符
            # 判断当前是否是文件
            if os.path.isfile(oldname):
                # 设置新文件名
                s = str(n+1).zfill(6)
                newname = newPath + os.sep + '2023516' + s + '.xml'  # 自定义新的文件名
                shutil.copy(oldname, newname)  # 复制图片并更改文件名  oldPath => newPath
                # os.rename(oldname, newname)  # 直接在源文件位置更改源文件名
                print(oldname, '======>', newname)

                n += 1
    else:
        print('路径不存在')

if __name__ == '__main__':
    oldPath = r"E:\tuberculosis-phonecamera\xml"  # 原图片存放路径
    newPath = r"E:\tuberculosis-phonecamera\new\xml"  # 更改名字后图片存放路径
    # os.remove(newPath)
    # os.makedirs(newPath)
    reName(oldPath, newPath)
