"""
MobileNetV3-Small 训练脚本（39类水果）
用于模型压缩部署，目标准确率 92%+
保存到独立目录，避免覆盖 ConvNeXtV2-Tiny 模型
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

# 覆盖配置 - MobileNetV3-Small
config.MODEL_NAME = "mobilenet_v3"
config.DATA_DIR = "./dataset"
config.IMAGE_SIZE = 224  # MobileNetV3 使用 224
config.BATCH_SIZE = 64   # MobileNetV3 更小，可以用更大批次
config.NUM_EPOCHS = 50
config.LEARNING_RATE = 0.001
config.FREEZE_BACKBONE = True
config.USE_AMP = True
config.NUM_WORKERS = 2
config.AUGMENTATION_MODE = "light"
config.USE_MIXUP = False
config.USE_CLASS_WEIGHTS = True
config.EARLY_STOP_PATIENCE = 10

# 保存到独立目录，避免覆盖 ConvNeXtV2-Tiny 模型
config.SAVE_DIR = "./checkpoints/mobilenetv3_small"

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("MobileNetV3-Small 训练 (39类水果)", flush=True)
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
    print(f"  保存目录: {config.SAVE_DIR}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("目标:", flush=True)
    print("  • 模型大小: ~10 MB", flush=True)
    print("  • 参数量: ~2.5M", flush=True)
    print("  • 目标准确率: 92%+", flush=True)
    print("  • 适合移动端部署", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    print("⚠️  注意: 模型保存到独立目录，不会覆盖 ConvNeXtV2-Tiny", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    train.main()
