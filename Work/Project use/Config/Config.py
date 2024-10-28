# -*- coding: utf-8 -*-
# @Time    : 2024/6/21 11:44
# @Author  : sjh
# @Site    : 
# @File    : config.py
# @Comment :
import os
import configparser
from pathlib import Path

class Config:
    project_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    save_json_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conf = None

    @classmethod
    def initialize(cls, filename='config.ini'):
        """读取配置文件并初始化类变量"""
        cls.filename = filename
        cls.conf = cls.read_config()

    @classmethod
    def get_source_path(cls):
        """项目总路径"""
        return cls.source_base_path

    @classmethod
    def get_sub_path(cls, sub_directory=None):
        """返回项目下的文件路径"""
        if isinstance(sub_directory, str):
            return os.path.join(cls.project_base_path, sub_directory)
        elif isinstance(sub_directory, list):
            return os.path.join(cls.project_base_path, *sub_directory)
        return cls.project_base_path

    @classmethod
    def get_save_json_path(cls, sub_directory=None):
        """返回存取json的路径"""
        if isinstance(sub_directory, str):
            return os.path.join(cls.save_json_path, sub_directory)
        elif isinstance(sub_directory, list):
            return os.path.join(cls.save_json_path, *sub_directory)
        return cls.save_json_path

    @classmethod
    def read_config(cls):
        """读取配置文件并返回ConfigParser对象"""
        config_path = Path(cls.source_base_path, 'resources/alg/config.ini')
        conf = configparser.ConfigParser()
        # 显式指定文件编码为 utf-8
        with open(config_path, 'r', encoding='utf-8') as fp:
            conf.read_file(fp)
        return conf

    @classmethod
    def get_config_param(cls, section, param):
        """获取特定的配置参数"""
        if cls.conf is None:
            raise Exception("Configuration not initialized. Call 'initialize' method first.")
        return cls.conf.get(section, param)


Config.initialize()
# 主函数示例
if __name__ == "__main__":
    # 初始化配置
    Config.initialize()

    # 获取特定配置项的值
    section = 'BodyMeasurement'
    param = 'parameterA'
    value = Config.get_config_param(section, param)

    print(f"Value of {param} in {section}: {value}")
    print(Config.get_source_path())
# # 示例用法
# if __name__ == "__main__":
#     # 使用示例
#     sub_directory_list = ['model_inference', 'onnx_inference.py']
#     path = Config.get_save_json_path()
#     print(path)  # 输出: /base/directory/sub1/sub2/sub3
#     print(Config.save_json_path)
