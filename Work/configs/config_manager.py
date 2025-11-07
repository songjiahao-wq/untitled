# -*- coding: utf-8 -*-
# @Time    : 2025/11/7 上午9:16
# @Author  : sjh
# @Site    : 
# @File    : config_manager.py
# @Comment :
import os
from pathlib import Path
from ruamel.yaml import YAML


class YamlConfig:
    """支持注释保留的 YAML 配置管理类"""

    def __init__(self, config_name="config.yaml", config_dir="configs"):
        # 自动获取项目根路径（即本文件所在项目根目录）
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / config_dir / config_name

        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ 配置文件未找到: {self.config_path}")

        self.yaml = YAML()
        self.yaml.preserve_quotes = True     # 保留引号
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.width = 4096               # 防止长行折行

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = self.yaml.load(f)

    # ------------------------------------------------------
    def get(self, key_path, default=None):
        """通过层级路径获取配置项，例如 'database.host'"""
        keys = key_path.split(".")
        value = self.data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path, value):
        """设置配置项，例如 'training.batch_size', 64"""
        keys = key_path.split(".")
        d = self.data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    # ------------------------------------------------------
    def save(self, backup=True):
        """写回文件，可自动备份原文件"""
        if backup:
            backup_path = self.config_path.with_suffix(".bak.yaml")
            os.replace(self.config_path, backup_path)
            print(f"🗂 已备份原配置到: {backup_path}")

        with open(self.config_path, "w", encoding="utf-8") as f:
            self.yaml.dump(self.data, f)
        print(f"✅ 配置已保存到: {self.config_path}")

    # ------------------------------------------------------
    def show(self):
        """打印当前配置内容"""
        import pprint
        print("🔧 当前配置:")
        pprint.pprint(self.data)

    def get_project_path(self, *subpath):
        """获取项目路径下的文件"""
        return str(self.project_root.joinpath(*subpath))
if __name__ == "__main__":

    cfg = YamlConfig()

    # 读取配置
    print(cfg.get("training.batch_size"))  # 输出 32

    # 修改配置
    cfg.set("training.batch_size", 64)
    cfg.set("database.host", "192.168.1.10")

    # 新增字段
    cfg.set("training.optimizer", "adam")

    # 打印当前配置
    cfg.show()

    # 保存并保留注释
    cfg.save()

    # 获取项目路径下的文件路径
    print(cfg.get_project_path("data", "train.csv"))
