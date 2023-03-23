# .txt-->.xml
# ! /usr/bin/python
# -*- coding:UTF-8 -*-
import os
import cv2


def txt_to_xml(txt_path, img_path, xml_path):
    # 1.字典对标签中的类别进行转换
    dict = {'0': "smoke",
}
    # 2.找到txt标签文件夹
    files = os.listdir(txt_path)
    # 用于存储 "老图"
    pre_img_name = ''
    # 3.遍历文件夹
    for i, name in enumerate(files):
        # 许多人文件夹里有该文件，默认的也删不掉，那就直接pass
        if name == "desktop.ini":
            continue
        print(name)
        # 4.打开txt
        txtFile = open(txt_path + name)
        # 读取所有内容
        txtList = txtFile.readlines()
        # 读取图片名称
        img_name = name.split(".")[0]
        pic = cv2.imread(img_path + img_name + ".jpg")
        # 获取图像大小信息
        Pheight, Pwidth, Pdepth = pic.shape
        # 5.遍历txt文件中每行内容
        for row in txtList:
            # 按' '分割txt的一行的内容
            oneline = row.strip().split(" ")
            # 遇到的是一张新图片
            if img_name != pre_img_name:
                # 6.新建xml文件
                xml_file = open((xml_path + img_name + '.xml'), 'w')
                xml_file.write('<annotation>\n')
                xml_file.write('    <folder>VOC2007</folder>\n')
                xml_file.write('    <filename>' + img_name + '.jpg' + '</filename>\n')
                xml_file.write('<source>\n')
                xml_file.write('<database>orgaquant</database>\n')
                xml_file.write('<annotation>organoids</annotation>\n')
                xml_file.write('</source>\n')
                xml_file.write('    <size>\n')
                xml_file.write('        <width>' + str(Pwidth) + '</width>\n')
                xml_file.write('        <height>' + str(Pheight) + '</height>\n')
                xml_file.write('        <depth>' + str(Pdepth) + '</depth>\n')
                xml_file.write('    </size>\n')
                xml_file.write('    <object>\n')
                xml_file.write('<name>' + dict[oneline[0]] + '</name>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(
                    int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(
                    int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(
                    int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(
                    int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
                xml_file.close()
                pre_img_name = img_name  # 将其设为"老"图
            else:  # 不是新图而是"老图"
                # 7.同一张图片，只需要追加写入object
                xml_file = open((xml_path + img_name + '.xml'), 'a')
                xml_file.write('    <object>\n')
                xml_file.write('<name>' + dict[oneline[0]] + '</name>\n')
                '''  按需添加这里和上面
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                '''
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(
                    int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(
                    int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(
                    int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(
                    int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
                xml_file.close()

        # 8.读完txt文件最后写入</annotation>
        xml_file1 = open((xml_path + pre_img_name + '.xml'), 'a')
        xml_file1.write('</annotation>')
        xml_file1.close()
    print("Done !")


# 修改成自己的文件夹 注意文件夹最后要加上/
txt_to_xml(r"F:\sjh\DATA2\xianyu\smoke_Phone\1smoke\train\labels/",
           r"F:\sjh\DATA2\xianyu\smoke_Phone\1smoke\train\images/",
           r"F:\sjh\DATA2\xianyu\smoke_Phone\1smoke\train\xml/")