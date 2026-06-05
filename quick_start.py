"""
快速开始指南 - 运行此文件查看使用说明
更新时间: 2026-06-05
"""

import os
import sys
from pathlib import Path


def print_banner():
    print("=" * 60)
    print("🍎 水果识别模型 - 快速开始指南")
    print("=" * 60)
    print("模型: ConvNeXtV2-Tiny (97.14% 测试准确率)")
    print("数据集: 39类水果 (39,986张图片)")
    print("=" * 60)


def check_environment():
    """检查环境配置"""
    print("\n📋 环境检查:")
    print("-" * 40)

    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"✅ Python版本: {python_version}")

    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")
        print("   请运行: pip install torch torchvision")
        return False

    # 检查其他依赖
    try:
        import torchvision
        print(f"✅ TorchVision版本: {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision未安装")

    try:
        from PIL import Image
        print(f"✅ Pillow已安装")
    except ImportError:
        print("❌ Pillow未安装")

    try:
        import matplotlib
        print(f"✅ Matplotlib已安装")
    except ImportError:
        print("❌ Matplotlib未安装")

    try:
        import timm
        print(f"✅ timm已安装 (用于ConvNeXtV2)")
    except ImportError:
        print("❌ timm未安装")

    return True


def check_dataset():
    """检查数据集状态"""
    print("\n📁 数据集检查:")
    print("-" * 40)

    dataset_dir = Path("dataset_merged")

    if dataset_dir.exists():
        train_dir = dataset_dir / "train"
        val_dir = dataset_dir / "val"
        test_dir = dataset_dir / "test"

        if train_dir.exists():
            classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
            print(f"✅ 训练集: {len(classes)} 个类别")

            # 统计图片数量
            total_train = sum(len(list(train_dir.glob(f"{cls}/*"))) for cls in classes)
            total_val = sum(len(list(val_dir.glob(f"{cls}/*"))) for cls in classes) if val_dir.exists() else 0
            total_test = sum(len(list(test_dir.glob(f"{cls}/*"))) for cls in classes) if test_dir.exists() else 0

            print(f"   训练集: {total_train} 张图片")
            print(f"   验证集: {total_val} 张图片")
            print(f"   测试集: {total_test} 张图片")
            print(f"   类别: {', '.join(classes[:10])}...")
    else:
        print("❌ dataset_merged 目录不存在")
        print("   请确保数据集已准备好")


def check_model():
    """检查模型状态"""
    print("\n🤖 模型检查:")
    print("-" * 40)

    model_file = Path("checkpoints/best_model.pth")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"✅ 最佳模型: checkpoints/best_model.pth ({size_mb:.1f} MB)")
    else:
        print("❌ 最佳模型不存在")

    # 检查训练信息
    info_file = Path("checkpoints/training_info_convnextv2_39class.json")
    if info_file.exists():
        print(f"✅ 训练信息: checkpoints/training_info_convnextv2_39class.json")


def show_next_steps():
    """显示下一步操作"""
    print("\n" + "=" * 60)
    print("📝 下一步操作:")
    print("=" * 60)

    print("""
1️⃣  使用模型预测 (单张图片)
   ────────────────────────────────────────────
   python tools/predict_single.py test.jpg

2️⃣  批量预测
   ────────────────────────────────────────────
   python tools/predict.py --dir ./test_images

3️⃣  实时摄像头识别
   ────────────────────────────────────────────
   python tools/camera_realtime.py

4️⃣  测试集评估
   ────────────────────────────────────────────
   python tools/evaluate_test.py

5️⃣  查看训练曲线 (TensorBoard)
   ────────────────────────────────────────────
   tensorboard --logdir=logs/tensorboard
   然后打开 http://localhost:6006

💡 提示：
   - 当前模型: ConvNeXtV2-Tiny (97.14% 测试准确率)
   - 支持 39 类水果识别
   - 摄像头实时识别: 28.9 FPS
   - 详细说明请查看 docs/ 目录下的文档
""")


def show_training_info():
    """显示训练信息"""
    print("\n" + "=" * 60)
    print("📊 训练信息:")
    print("=" * 60)

    print("""
模型: ConvNeXtV2-Tiny
  - 总参数量: 28,280,231
  - 预训练权重: ImageNet-22K
  - 输入尺寸: 288×288

训练策略: 两阶段训练
  - 阶段1: 冻结训练 (21轮, 94.82%)
  - 阶段2: 微调 (30轮, 97.03%)

测试结果:
  - 测试准确率: 97.14%
  - 平均精确率: 96.27%
  - 平均召回率: 97.08%
  - 平均F1值: 96.60%

13个类别达到100%准确率:
  - 蓝莓、杨桃、蔓越莓、榴莲、葡萄
  - 猕猴桃、金桔、柠檬、枇杷、桑葚
  - 梨、菠萝、番茄
""")


def main():
    print_banner()

    env_ok = check_environment()
    check_dataset()
    check_model()

    show_next_steps()
    show_training_info()

    if not env_ok:
        print("\n⚠️ 请先安装依赖：")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
