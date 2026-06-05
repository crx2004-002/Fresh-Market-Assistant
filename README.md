# 🍎 水果识别模型 (Fresh Market Assistant)

基于深度学习的水果图像分类项目，使用 ConvNeXtV2-Tiny 模型实现 39 类水果识别。

## 📊 项目成果

- **测试准确率**: 97.14%
- **验证准确率**: 97.03%
- **支持类别**: 39 类水果
- **摄像头实时识别**: 28.9 FPS
- **模型大小**: ~324 MB

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 如果需要GPU加速，请到PyTorch官网获取对应CUDA版本的安装命令
# https://pytorch.org/get-started/locally/
```

### 2. 使用模型预测

```bash
# 预测单张图片
python tools/predict_single.py test.jpg

# 批量预测
python tools/predict.py --dir ./test_images

# 实时摄像头识别
python tools/camera_realtime.py
```

### 3. 测试集评估

```bash
python tools/evaluate_test.py
```

### 4. 查看训练曲线

```bash
tensorboard --logdir=logs/tensorboard
# 然后打开 http://localhost:6006
```

## 📁 项目结构

```
Fresh Market Assistant/
├── checkpoints/                    # 模型文件
│   ├── best_model.pth             # 最佳模型 (97.03%)
│   ├── last_model.pth             # 最后一轮模型
│   └── training_info_convnextv2_39class.json
├── dataset_merged/                 # 数据集 (39类水果)
│   ├── train/                     # 训练集 (31,648张)
│   ├── val/                       # 验证集 (3,871张)
│   └── test/                      # 测试集 (3,852张)
├── docs/                          # 文档
│   ├── 训练报告.md                 # 完整训练报告
│   ├── 迁移学习与微调完整训练日志.md
│   └── 训练说明.md
├── logs/                          # 日志和图片
│   ├── training_images/           # 导出的训练图片
│   ├── tensorboard/               # TensorBoard日志
│   └── training_report_convnextv2_39class.md
├── tools/                         # 工具脚本
│   ├── model_factory.py           # 模型工厂
│   ├── evaluate_test.py           # 测试评估
│   ├── camera_realtime.py         # 实时摄像头识别
│   ├── predict_single.py          # 单张图片预测
│   ├── predict.py                 # 批量预测
│   ├── debug_camera.py            # 摄像头调试
│   └── class_names_cn.py          # 中英文名称映射
├── config.py                      # 配置文件
├── train.py                       # 主训练脚本
├── train_convnextv2.py            # ConvNeXtV2训练脚本
├── train_convnextv2_finetune.py   # ConvNeXtV2微调脚本
├── quick_start.py                 # 快速开始指南
├── requirements.txt               # 依赖包列表
└── README.md                      # 本文件
```

## ⚙️ 模型配置

### ConvNeXtV2-Tiny (当前模型)

| 配置项 | 值 |
|--------|-----|
| 模型架构 | ConvNeXtV2-Tiny |
| 预训练权重 | ImageNet-22K |
| 总参数量 | 28,280,231 |
| 输入尺寸 | 288×288 |
| 输出类别 | 39 |

### 训练配置

| 配置项 | 冻结训练 | 微调 |
|--------|----------|------|
| 学习率 | 0.001 | 0.0001 |
| 批次大小 | 32 | 16 |
| 训练轮数 | 21 | 30 |
| 冻结骨干 | 是 | 否 |
| 验证准确率 | 94.82% | 97.03% |

## 📊 支持的水果类别 (39类)

```
apple, apricot, avocado, banana, bayberry, blueberry, cantaloupe, carambola,
cherry, coconut, cranberry, dragonfruit, durian, fig, grape, grapefruit,
guava, hawthorn, jackfruit, kiwi fruit, kumquat, lemon, longan, loquat,
lychee, mandarine, mango, mulberry, orange, peach, pear, persimmon,
pineapple, plumcot, pomegranate, pomelo, strawberry, tomato, watermelon
```

## 📈 训练历程

### 阶段1: 迁移学习 (冻结训练)
- 训练轮次: 21轮
- 训练时间: ~23.8小时
- 最佳验证准确率: 94.82%

### 阶段2: 微调
- 训练轮次: 30轮
- 训练时间: ~4.8小时
- 最佳验证准确率: 97.03%

### 测试结果
- 测试准确率: 97.14%
- 平均精确率: 96.27%
- 平均召回率: 97.08%
- 平均F1值: 96.60%

## 🏆 100%准确率的类别 (13个)

- 蓝莓 (blueberry)
- 杨桃 (carambola)
- 蔓越莓 (cranberry)
- 榴莲 (durian)
- 葡萄 (grape)
- 猕猴桃 (kiwi fruit)
- 金桔 (kumquat)
- 柠檬 (lemon)
- 枇杷 (loquat)
- 桑葚 (mulberry)
- 梨 (pear)
- 菠萝 (pineapple)
- 番茄 (tomato)

## 🔧 进阶使用

### 修改训练参数

编辑 `config.py` 文件：

```python
# 模型配置
MODEL_NAME = "convnextv2_tiny"
IMAGE_SIZE = 288
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 训练配置
NUM_EPOCHS = 50
FREEZE_BACKBONE = True
USE_AMP = True
```

### 重新训练

```bash
# 冻结训练
python train_convnextv2.py

# 微调
python train_convnextv2_finetune.py
```

## 📚 文档

- [训练报告](docs/训练报告.md) - 完整的训练报告
- [迁移学习与微调完整训练日志.md](docs/迁移学习与微调完整训练日志.md) - 详细训练日志
- [训练说明.md](docs/训练说明.md) - 训练使用指南

## 📝 训练输出示例

```
============================================================
ConvNeXtV2-Tiny 训练 (39类水果)
============================================================
  模型: convnextv2_tiny
  数据集: ./dataset_merged
  图片尺寸: 288
  批次大小: 32
  训练轮数: 50
  学习率: 0.001
  冻结骨干: True
  AMP混合精度: True
  数据增强: light
============================================================

[TRAIN] Epoch [  1/50] Train Loss: 1.8384 Acc: 0.6371 | Val Loss: 1.3915 Acc: 0.7945 | LR: 0.000999 | Time: 1510.5s
[TRAIN] Epoch [  2/50] Train Loss: 1.4682 Acc: 0.7574 | Val Loss: 1.2465 Acc: 0.8506 | LR: 0.000996 | Time: 1548.7s
...

✅ 训练完成! 总耗时: 23.8小时
   最佳验证准确率: 0.9482
============================================================
```

## 📄 License

本项目仅供学习使用。

---

**更新时间**: 2026-06-05
**模型版本**: ConvNeXtV2-Tiny v1.0
**训练环境**: conda yolov (Python 3.12.13, PyTorch 2.0+, CUDA)
