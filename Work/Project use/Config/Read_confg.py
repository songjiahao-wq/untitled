# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 15:10
# @Author  : sjh
# @Site    : 
# @File    : Read_confg.py
# @Comment :
import os
import configparser

# 读取配置文件并返回字典
def Read_config(filename):
    # 获取当前目录路径
    proDir = os.path.split(os.path.realpath(__file__))[0]
    # 拼接路径获取完整路径
    configPath = os.path.join(proDir, filename)
    # 创建ConfigParser对象
    conf = configparser.ConfigParser()
    # 读取文件内容
    conf.read(configPath)
    # # 获取所有配置项
    # config_dict = {}
    # for section in conf.sections():
    #     config_dict[section] = {}
    #     for option in conf.options(section):
    #         config_dict[section][option] = conf.get(section, option)
    return conf
#
# #
# # 调用函数读取配置文件
# config_dict = read_config('config.ini')
# print(config_dict.get('path','root_path'))
