# 计算机视觉项目集合

这是一个包含多个计算机视觉相关功能的项目仓库，主要涵盖注意力机制、数据集处理、目标检测等领域。

## 📁 项目结构

### Attention/
注意力机制相关模块，包含多种先进的注意力机制实现：
- **ComplexDualAttentionFusion.py** - 复杂双注意力融合
- **CAConv.py / CAConv2.py** - 坐标注意力卷积
- **AFF.py** - 注意力特征融合
- **GAM_Attention.py** - GAM注意力机制
- **FusionModel.py** - 特征融合模型
- **EPSA.py / EPSA+SPA.py** - 高效空间注意力
- **EMA_ATT.py** - EMA注意力
- **SoftThresholdAttention.py** - 软阈值注意力
- **频域注意力.py** - 频域注意力机制
- **去噪注意力.py / 去噪.py** - 去噪相关注意力

以及其他先进的网络架构：
- **MobielVit.py** - MobileViT架构
- **InceptionNext.py** - InceptionNext网络
- **Ghostnetv2.py** - GhostNetV2
- **SCConv.py** - 空间通道卷积
- **convnextv2.py** - ConvNeXtV2
- **backbone_main.py** - 主干网络

### youhua/
数据处理和优化相关工具：
- **数据集相关/** - 数据格式转换和处理
  - VOC格式转换
  - YOLO格式转换
  - XML/TXT/JSON互转
- **数据增多/** - 数据增强技术
  - Copy-Paste增强
  - 语义分割增强
- **安全区域检测/** - 安全区域检测算法
- **切图/** - 图像切割工具
- **分割/** - 图像分割相关
- **三钢/** - 特定数据集处理
- **widerperson/** - WiderPerson数据集处理
- **INRIAPerson/** - INRIA Person数据集处理
- **NEU数据集/** - NEU数据集处理
- **crowdhuman/** - CrowdHuman数据集处理
- **kaggle/** - Kaggle相关

### other/
其他计算机视觉应用：
- **Face_Dection.py** - 人脸检测
- **Virtual_Drag_and_Drop.py** - 虚拟拖拽
- **Virtual_Quiz_Game.py** - 虚拟问答游戏
- **mediaPipe.py** - MediaPipe应用
- **opencv_face_det.py** - OpenCV人脸检测
- **cvzone_face_dect.py / cvzone_facemesh_det.py** - CVZone相关
- **dlib_matching.py** - Dlib匹配

### 工具模块
- **Yolo-to-COCO-format-converter-master/** - YOLO到COCO格式转换器
- **cifar-10/** - CIFAR-10数据集处理
- **yolo2coco/** - YOLO转COCO工具

## 🚀 主要功能

### 1. 注意力机制
- 实现了多种先进的注意力机制
- 支持即插即用到现有网络中
- 包含空间注意力、通道注意力、频域注意力等

### 2. 数据集处理
- 支持多种数据格式转换（VOC、YOLO、COCO、JSON）
- 数据集划分和验证
- 标注格式标准化
- 数据质量检查和修复

### 3. 数据增强
- Copy-Paste数据增强
- 语义分割增强
- 图像切割和重组
- 自动化数据增广流程

### 4. 目标检测
- 多种目标检测算法实现
- 人脸检测和识别
- 安全区域检测
- 人体检测相关算法

## 📋 依赖环境

推荐使用Python 3.7+，主要依赖包括：

```bash
# 计算机视觉基础库
opencv-python
pillow
scikit-image

# 深度学习框架
torch
torchvision
tensorflow

# 数据处理
numpy
pandas
matplotlib

# 工具库
tqdm
xmltodict
pyyaml
```

## 🛠️ 使用说明

### 注意力机制使用
```python
from Attention.CAConv import CAConv
from Attention.GAM_Attention import GAM_Attention

# 创建注意力模块
attention = CAConv(in_channels=64, out_channels=64)
gam = GAM_Attention(in_channels=64, out_channels=64)

# 在网络中使用
output = attention(input_tensor)
```

### 数据格式转换
```python
# YOLO转COCO
python Yolo-to-COCO-format-converter-master/main.py

# XML转TXT
python youhua/数据集相关/xml2txt.py

# 数据集划分
python youhua/数据集相关/yolo2train-test-val.py
```

### 数据增强
```python
# Copy-Paste增强
python youhua/数据增多/copy-paste.py

# 语义分割增强
python youhua/数据增多/语义分割.py
```

## 📊 项目特点

1. **模块化设计** - 各功能模块独立，便于维护和扩展
2. **多格式支持** - 支持主流数据格式和标注格式
3. **即插即用** - 注意力模块可直接集成到现有项目中
4. **完整工具链** - 从数据处理到模型训练的完整工具支持
5. **实用性强** - 包含多个实际应用场景的解决方案

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证，详情请查看LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

*最后更新：2025年11月*