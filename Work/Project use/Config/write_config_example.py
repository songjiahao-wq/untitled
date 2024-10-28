# -*- coding: utf-8 -*-
# @Time    : 2024/8/2 14:39
# @Author  : sjh
# @Site    : 
# @File    : write_config_example.py
# @Comment :
from configobj import ConfigObj

# 定义配置文件路径
config_file_path = '../../../resources/alg/config.ini'

# 读取配置文件，如果不存在则创建新的
config = ConfigObj(config_file_path, encoding='utf-8')

# 定义需要更新或添加的配置内容
new_config = {
    'kinect_ui2': {
        'FRAME_WIDTH': '1920',
        'FRAME_HEIGHT': '1080'
    },
    'BodyMeasurement': {
        'parameterA': 'valueA',
        'frame_count': '50'
    }
}

# 更新或添加配置内容
for section, params in new_config.items():
    if section not in config:
        config[section] = {}
    for key, value in params.items():
        config[section][key] = value

# 将更新后的配置写入 config.ini 文件
config.write()

print(f"Configuration updated and saved to {config_file_path}")
def get_camera_param(kinect):
    from configobj import ConfigObj
    from pathlib import Path
    from .Config import Config

    # 定义配置文件路径
    config_file_path = Path(Config.get_source_path()) / Path('resources/Open3d_port/config/config.ini')

    # 读取配置文件，如果不存在则创建新的
    config = ConfigObj(str(config_file_path), encoding='utf-8')

    intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
    focal_length_x = intrinsics_matrix.FocalLengthX
    focal_length_y = intrinsics_matrix.FocalLengthY
    principal_point_x = intrinsics_matrix.PrincipalPointX
    principal_point_y = intrinsics_matrix.PrincipalPointY
    radial_distortion_fourth_order = intrinsics_matrix.RadialDistortionFourthOrder
    radial_distortion_second_order = intrinsics_matrix.RadialDistortionSecondOrder
    radial_distortion_sixth_order = intrinsics_matrix.RadialDistortionSixthOrder
    print(f"相机的焦距xy{focal_length_x, focal_length_y},相机的主点位置{principal_point_x, principal_point_y},"
          f"径向畸变系数{radial_distortion_fourth_order, radial_distortion_second_order, radial_distortion_sixth_order}")

    # 定义需要更新或添加的配置内容
    new_config = {
        'kinect_camera_config': {
            'fx': round(focal_length_x, 3),
            'fy': round(focal_length_y, 3),
            'cx': round(principal_point_x, 3),
            'cy': round(principal_point_y,3)
        },

    }
    # 更新或添加配置内容
    for section, params in new_config.items():
        if section not in config:
            config[section] = {}
        for key, value in params.items():
            config[section][key] = value
    # 将更新后的配置写入 config.ini 文件
    config.write()

    print(f"Configuration updated and saved to {config_file_path}")