"""
ConvNeXtV2-Tiny 训练脚本（39类水果）
配置 HuggingFace 镜像源，解决国内下载问题
强制实时输出训练进度
"""
import os
import sys

# 强制 Python 不缓冲输出（实时显示训练进度）
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 强制 stdout 不缓冲
sys.stdout.reconfigure(line_buffering=True)

import config
import train

# 覆盖配置
config.MODEL_NAME = "convnextv2_tiny"
config.DATA_DIR = "./dataset"
config.IMAGE_SIZE = 288
config.BATCH_SIZE = 32
config.NUM_EPOCHS = 50
config.LEARNING_RATE = 0.001
config.FREEZE_BACKBONE = True
config.USE_AMP = True
config.NUM_WORKERS = 2
config.AUGMENTATION_MODE = "light"
config.USE_MIXUP = False
config.USE_CLASS_WEIGHTS = True

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("ConvNeXtV2-Tiny 训练 (39类水果)", flush=True)
    print("=" * 60, flush=True)
    print(f"  模型: {config.MODEL_NAME}", flush=True)
    print(f"  数据集: {config.DATA_DIR}", flush=True)
    print(f"  图片尺寸: {config.IMAGE_SIZE}", flush=True)
    print(f"  批次大小: {config.BATCH_SIZE}", flush=True)
    print(f"  训练轮数: {config.NUM_EPOCHS}", flush=True)
    print(f"  学习率: {config.LEARNING_RATE}", flush=True)
    print(f"  冻结骨干: {config.FREEZE_BACKBONE}", flush=True)
    print(f"  AMP混合精度: {config.USE_AMP}", flush=True)
    print(f"  数据增强: {config.AUGMENTATION_MODE}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    train.main()
