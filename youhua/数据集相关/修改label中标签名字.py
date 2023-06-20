import xml.etree.ElementTree as ET
import os

new_label = 2
def reLabelName(old_xml_path, new_xml_path, newlabel):
    # 判断路径是否存在
    if os.path.exists(old_xml_path):
        # 获取该目录下所有文件，存入列表中
        fileList = os.listdir(old_xml_path)
        if len(fileList) > 0:
            if not os.path.exists(new_xml_path):
                os.makedirs(new_xml_path)
        for xml in fileList:
            old_xml_full_path = os.path.join(old_xml_path, xml)
            tree = ET.parse(old_xml_full_path)  # 解析xml文件路径
            nodes = tree.findall('./object/name')
            for node in nodes:
                node.text = new_label
            new_xml_full_path = os.path.join(new_xml_path, xml)
            tree.write(new_xml_full_path)

if __name__ == '__main__':
    old_xml_path = 'F:\\2021-08-20\labels'
    new_xml_path = 'F:\\2021-08-20\labels1'
    ne_label = 'paraleyrodes_pseudonaranjae_martin'
    reLabelName(old_xml_path, new_xml_path, ne_label)
