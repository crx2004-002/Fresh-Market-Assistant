"""
MobileNetV3-Small 微调脚本（39类水果）
在冻结训练基础上解冻骨干，提升准确率到 92%+
"""
import os
import sys

# 强制 Python 不缓冲输出
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 强制 stdout 不缓冲
sys.stdout.reconfigure(line_buffering=True)

import config
import train

# 覆盖配置 - MobileNetV3-Small 微调
config.TRAIN_MODE = "finetune"
config.MODEL_NAME = "mobilenet_v3"
config.DATA_DIR = "./dataset"
config.IMAGE_SIZE = 224
config.BATCH_SIZE = 64
config.NUM_EPOCHS = 30
config.LEARNING_RATE = 0.0001  # 微调使用更小学习率
config.FREEZE_BACKBONE = False  # 解冻骨干
config.FINETUNE_MODEL_PATH = "./checkpoints/mobilenetv3_small/best_model.pth"
config.FINETUNE_LEARNING_RATE = 0.0001
config.FINETUNE_NUM_EPOCHS = 30
config.FINETUNE_UNFREEZE_LEVEL = -1  # 全部解冻
config.USE_AMP = True
config.NUM_WORKERS = 2
config.AUGMENTATION_MODE = "light"
config.USE_MIXUP = False
config.USE_CLASS_WEIGHTS = True
config.EARLY_STOP_PATIENCE = 10

# 保存到独立目录
config.SAVE_DIR = "./checkpoints/mobilenetv3_small"

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("MobileNetV3-Small 微调 (39类水果)", flush=True)
    print("=" * 60, flush=True)
    print(f"  模型: {config.MODEL_NAME}", flush=True)
    print(f"  数据集: {config.DATA_DIR}", flush=True)
    print(f"  图片尺寸: {config.IMAGE_SIZE}", flush=True)
    print(f"  批次大小: {config.BATCH_SIZE}", flush=True)
    print(f"  训练轮数: {config.NUM_EPOCHS}", flush=True)
    print(f"  学习率: {config.LEARNING_RATE}", flush=True)
    print(f"  冻结骨干: {config.FREEZE_BACKBONE}", flush=True)
    print(f"  解冻级别: {config.FINETUNE_UNFREEZE_LEVEL} (全部解冻)", flush=True)
    print(f"  基础模型: {config.FINETUNE_MODEL_PATH}", flush=True)
    print(f"  AMP混合精度: {config.USE_AMP}", flush=True)
    print(f"  数据增强: {config.AUGMENTATION_MODE}", flush=True)
    print(f"  保存目录: {config.SAVE_DIR}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("目标:", flush=True)
    print("  • 基础准确率: 91.10%", flush=True)
    print("  • 目标准确率: 92%+", flush=True)
    print("  • 预期提升: 1-3%", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    train.main()
