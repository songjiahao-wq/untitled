# -*- coding: utf-8 -*-
# @Time    : 2024/6/21 16:08
# @Author  : sjh
# @Site    : 
# @File    : 111.py
# @Comment :
from Config import Config
from pathlib import Path
import os
# 获取特定配置项的值
section = 'kinect_camera_config'
param = 'fx'
value = Config.get_config_param(section, param)
a = ('kinect_camera_config', 'fx')

print(float(value))


source = Config.get_source_path()
a = Path(source)/'alg/Open3d_port/config'
print(type(Path(source)/'alg/Open3d_port/config'))
print(os.path.join(a,'config.json'))