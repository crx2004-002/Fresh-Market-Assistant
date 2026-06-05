"""
ConvNeXtV2-Tiny 微调脚本（39类水果）
在冻结骨干训练的基础上，解冻骨干进行精细调整
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

# 覆盖配置 - 微调模式
config.TRAIN_MODE = "finetune"
config.MODEL_NAME = "convnextv2_tiny"
config.DATA_DIR = "./dataset"
config.IMAGE_SIZE = 288
config.BATCH_SIZE = 16  # 微调时减小批次，因为需要更多显存
config.NUM_EPOCHS = 30
config.LEARNING_RATE = 0.0001  # 微调使用更小的学习率
config.FREEZE_BACKBONE = False  # 解冻骨干网络
config.FINETUNE_MODEL_PATH = "./checkpoints/best_model.pth"
config.FINETUNE_LEARNING_RATE = 0.0001
config.FINETUNE_NUM_EPOCHS = 30
config.FINETUNE_UNFREEZE_LEVEL = -1  # 全部解冻
config.USE_AMP = True
config.NUM_WORKERS = 2
config.AUGMENTATION_MODE = "light"
config.USE_MIXUP = False
config.USE_CLASS_WEIGHTS = True
config.EARLY_STOP_PATIENCE = 10

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("ConvNeXtV2-Tiny 微调 (39类水果)", flush=True)
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
    print("=" * 60, flush=True)
    print(flush=True)
    print("微调说明:", flush=True)
    print("  • 解冻骨干网络，训练所有参数", flush=True)
    print("  • 使用更小的学习率 (0.0001)", flush=True)
    print("  • 减小批次大小 (16) 以节省显存", flush=True)
    print("  • 预期提升: 1-3%", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    train.main()
