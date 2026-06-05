"""
水果识别模型 - 训练脚本
使用PyTorch微调预训练模型进行水果分类
支持 2024 新模型: DINOv2, ConvNeXtV2, EfficientViT
支持 AMP 混合精度训练
"""

import os
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免卡死
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 查找可用的中文字体
def find_chinese_font():
    """查找系统中可用的中文字体"""
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
        'C:/Windows/Fonts/simhei.ttf',     # 黑体
        'C:/Windows/Fonts/simsun.ttc',     # 宋体
        'C:/Windows/Fonts/STKAITI.TTF',    # 华文楷体
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return fp
    return None

# 配置中文字体
chinese_font_path = find_chinese_font()
if chinese_font_path:
    fm.fontManager.addfont(chinese_font_path)
    font_prop = fm.FontProperties(fname=chinese_font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    # 如果没有中文字体，使用英文
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False

# TensorBoard（使用 tensorboardX，无 TensorFlow 依赖）
HAS_TENSORBOARD = False
try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    pass
from datetime import datetime

import config
from tools.model_factory import (
    create_model, get_train_transforms, get_val_transforms,
    freeze_backbone, apply_gradual_unfreezing, get_default_image_size
)

# 限制PyTorch CPU线程数，防止CPU占满导致系统卡死
torch.set_num_threads(2)


def set_seed(seed):
    """设置随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取训练设备"""
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[GPU] 使用CPU训练（会比较慢）")
    return device


def get_transforms():
    """
    获取数据预处理/增强
    训练集使用数据增强提高泛化能力
    验证集/测试集只做标准化
    """
    aug_mode = getattr(config, 'AUGMENTATION_MODE', 'trivial')
    train_transform = get_train_transforms(config.IMAGE_SIZE, mode=aug_mode)
    val_test_transform = get_val_transforms(config.IMAGE_SIZE)
    return train_transform, val_test_transform


def load_datasets():
    """加载数据集"""
    train_transform, val_test_transform = get_transforms()

    data_dir = Path(config.DATA_DIR)

    # 检查数据集是否存在
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在: {data_dir}\n"
            f"请先运行 prepare_dataset.py 准备数据集"
        )

    # 加载数据集
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_test_transform)

    # 获取类别信息
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\n[INFO] 数据集信息:")
    print(f"   类别数: {num_classes}")
    print(f"   类别: {', '.join(class_names)}")
    print(f"   训练集: {len(train_dataset)} 张")
    print(f"   验证集: {len(val_dataset)} 张")
    print(f"   测试集: {len(test_dataset)} 张")

    # Windows下pin_memory=False更稳定，避免内存锁定导致卡死
    use_pin_memory = config.USE_GPU and torch.cuda.is_available()

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader, class_names, num_classes


def create_training_model(num_classes, load_from_checkpoint=False):
    """
    创建模型（使用统一模型工厂）
    Args:
        num_classes: 分类数
        load_from_checkpoint: 是否从检查点加载模型权重（用于二次微调）
    """
    print(f"\n[MODEL] 加载预训练模型: {config.MODEL_NAME}")

    model = create_model(config.MODEL_NAME, num_classes, pretrained=not load_from_checkpoint)

    # 冻结骨干网络（首次训练时）
    if config.FREEZE_BACKBONE and not load_from_checkpoint:
        freeze_backbone(model, config.MODEL_NAME)

    # 从检查点加载权重（二次微调）
    if load_from_checkpoint:
        checkpoint_path = config.FINETUNE_MODEL_PATH
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"微调模型不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   [OK] 已加载微调模型: {checkpoint_path}")
        print(f"      原始验证准确率: {checkpoint.get('val_acc', 'N/A')}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    return model


def apply_gradual_unfreezing_local(model, unfreeze_level):
    """渐进式解冻策略（委托给 model_factory）"""
    return apply_gradual_unfreezing(model, config.MODEL_NAME, unfreeze_level)


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None, accumulation_steps=1, mixup_fn=None):
    """
    训练一个epoch（支持 AMP 混合精度 + 梯度累积 + Mixup/CutMix）
    Args:
        accumulation_steps: 梯度累积步数，等效 batch = real_batch * accumulation_steps
        mixup_fn: Mixup/CutMix 函数
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup/CutMix
        if mixup_fn is not None:
            images, labels_a, labels_b, lam = mixup_fn(images, labels, len(train_loader.dataset.classes))

        if scaler is not None:
            # AMP 混合精度训练
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                if mixup_fn is not None:
                    loss = (lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)) / accumulation_steps
                else:
                    loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            # 标准训练
            outputs = model(images)
            if mixup_fn is not None:
                loss = (lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)) / accumulation_steps
            else:
                loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

        # 梯度累积：每 accumulation_steps 步更新一次
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # 统计（用原始 loss，不是除以累积步数的 loss）
        running_loss += loss.item() * accumulation_steps * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        if mixup_fn is not None:
            correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
        else:
            correct += (predicted == labels).sum().item()

        # 释放中间变量显存
        del images, labels, outputs, loss

    # 处理最后不足 accumulation_steps 的步数
    if (step + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 释放中间变量
            del images, labels, outputs, loss

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_history(history, save_path):
    """绘制训练曲线"""
    # 判断是否使用中文
    use_chinese = chinese_font_path is not None

    if use_chinese:
        labels = {
            'train_loss': '训练损失',
            'val_loss': '验证损失',
            'train_acc': '训练准确率',
            'val_acc': '验证准确率',
            'loss_title': '损失变化曲线',
            'acc_title': '准确率变化曲线',
            'epoch': '轮次',
            'loss': '损失',
            'acc': '准确率'
        }
    else:
        labels = {
            'train_loss': 'Train Loss',
            'val_loss': 'Val Loss',
            'train_acc': 'Train Acc',
            'val_acc': 'Val Acc',
            'loss_title': 'Loss Curve',
            'acc_title': 'Accuracy Curve',
            'epoch': 'Epoch',
            'loss': 'Loss',
            'acc': 'Accuracy'
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label=labels['train_loss'])
    ax1.plot(epochs, history['val_loss'], 'r-', label=labels['val_loss'])
    ax1.set_title(labels['loss_title'])
    ax1.set_xlabel(labels['epoch'])
    ax1.set_ylabel(labels['loss'])
    ax1.legend()
    ax1.grid(True)

    # Accuracy曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label=labels['train_acc'])
    ax2.plot(epochs, history['val_acc'], 'r-', label=labels['val_acc'])
    ax2.set_title(labels['acc_title'])
    ax2.set_xlabel(labels['epoch'])
    ax2.set_ylabel(labels['acc'])
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[CHART] 训练曲线已保存: {save_path}")


def save_training_info(history, class_names, save_dir):
    """保存训练信息"""
    info = {
        "model_name": config.MODEL_NAME,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": config.IMAGE_SIZE,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs": config.NUM_EPOCHS,
        "best_val_acc": max(history['val_acc']),
        "best_epoch": history['val_acc'].index(max(history['val_acc'])) + 1,
        "training_time": history.get('total_time', 'N/A'),
        "use_amp": getattr(config, 'USE_AMP', False),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    info_path = save_dir / "training_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[LOG] 训练信息已保存: {info_path}")


def main():
    """主训练流程"""
    # 判断训练模式
    is_finetune = config.TRAIN_MODE == "finetune"

    if is_finetune:
        print("=" * 60)
        print("水果识别模型 - 二次微调训练")
        print("=" * 60)
        print(f"   微调模型: {config.FINETUNE_MODEL_PATH}")
        print(f"   解冻级别: {config.FINETUNE_UNFREEZE_LEVEL}")
    else:
        print("=" * 60)
        print("水果识别模型 - 首次训练")
        print("=" * 60)

    # 设置随机种子
    set_seed(config.RANDOM_SEED)

    # 获取设备
    device = get_device()

    # 创建保存目录
    save_dir = Path(config.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard（使用不带中文的路径）
    writer = None
    if HAS_TENSORBOARD:
        # 使用临时目录避免中文路径问题
        import tempfile
        tb_dir = Path(tempfile.gettempdir()) / 'fruit_tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"   [OK] TensorBoard 日志: {tb_dir}")
        print(f"   启动命令: tensorboard --logdir={tb_dir}")
    else:
        print("   [WARN] TensorBoard 未安装，运行: pip install tensorboardX")

    # 加载数据集
    train_loader, val_loader, test_loader, class_names, num_classes = load_datasets()

    # 创建模型
    model = create_training_model(num_classes, load_from_checkpoint=is_finetune)
    model = model.to(device)

    # 二次微调：应用渐进式解冻
    if is_finetune:
        model = apply_gradual_unfreezing_local(model, config.FINETUNE_UNFREEZE_LEVEL)

    # 梯度检查点（节省显存，DINOv2 等大模型专用）
    use_grad_checkpoint = getattr(config, 'GRADIENT_CHECKPOINTING', False)
    if use_grad_checkpoint and hasattr(model, 'backbone'):
        try:
            model.backbone.set_grad_checkpointing()
            print("   [OK] 已启用梯度检查点")
        except Exception:
            pass

    # 类别权重（处理不平衡数据）
    use_class_weights = getattr(config, 'USE_CLASS_WEIGHTS', False)
    if use_class_weights:
        # 统计各类别样本数
        class_counts = []
        for cls in class_names:
            cls_dir = Path(config.DATA_DIR) / "train" / cls
            count = len(list(cls_dir.iterdir()))
            class_counts.append(count)
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        # 权重 = 总样本数 / (类别数 * 该类样本数)
        weights = class_counts.sum() / (len(class_counts) * class_counts)
        weights = weights / weights.min()  # 归一化，最小权重为1
        weights = weights.to(device)
        print(f"   类别权重:")
        for cls, w in zip(class_names, weights.cpu().numpy()):
            print(f"     {cls}: {w:.2f}")
    else:
        weights = None

    # 损失函数（支持类别权重和 Label Smoothing）
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing
    )
    if label_smoothing > 0:
        print(f"   Label Smoothing: {label_smoothing}")

    # 优化器
    learning_rate = config.FINETUNE_LEARNING_RATE if is_finetune else config.LEARNING_RATE
    weight_decay = getattr(config, 'WEIGHT_DECAY', 0.01)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"   Weight Decay: {weight_decay}")

    # AMP 混合精度训练
    use_amp = getattr(config, 'USE_AMP', False) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("   [OK] 已启用 AMP 混合精度训练")

    # 训练轮数
    num_epochs = config.FINETUNE_NUM_EPOCHS if is_finetune else config.NUM_EPOCHS

    # 学习率调度器
    if config.SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif config.SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config.SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    else:
        scheduler = None

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Mixup / CutMix 数据增强
    use_mixup = getattr(config, 'USE_MIXUP', False)
    if use_mixup:
        mixup_alpha = getattr(config, 'MIXUP_ALPHA', 1.0)
        cutmix_alpha = getattr(config, 'CUTMIX_ALPHA', 1.0)
        mixup_prob = getattr(config, 'MIXUP_PROB', 0.5)
        print(f"   [OK] 已启用 Mixup/CutMix (prob={mixup_prob})")

        def mixup_cutmix(images, labels, num_classes):
            """Mixup 或 CutMix 随机选择"""
            import random
            if random.random() > mixup_prob:
                return images, labels, labels, 1.0

            # 随机选择 Mixup 或 CutMix
            if random.random() < 0.5:
                # Mixup
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(images.device)
                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a = labels
                labels_b = labels[index]
                return mixed_images, labels_a, labels_b, lam
            else:
                # CutMix
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(images.device)

                # 生成裁剪区域
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

                # 调整 lambda
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

                labels_a = labels
                labels_b = labels[index]
                return images, labels_a, labels_b, lam

        def rand_bbox(size, lam):
            """生成 CutMix 裁剪区域"""
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2

    # 早停相关
    best_val_loss = float('inf')
    best_val_acc = 0.0  # 跟踪最佳验证准确率
    patience_counter = 0

    # 开始训练
    print(f"\n[START] 开始{'微调' if is_finetune else '训练'}...", flush=True)
    print(f"   轮次: {num_epochs}", flush=True)
    print(f"   批次大小: {config.BATCH_SIZE}", flush=True)
    print(f"   学习率: {learning_rate}", flush=True)
    print(f"   图像尺寸: {config.IMAGE_SIZE}", flush=True)
    print(f"   数据增强: {getattr(config, 'AUGMENTATION_MODE', 'trivial')}", flush=True)
    print(f"   类别权重: {'开启' if use_class_weights else '关闭'}", flush=True)
    print(f"   Mixup/CutMix: {'开启' if use_mixup else '关闭'}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 梯度累积步数
        accum_steps = getattr(config, 'ACCUMULATION_STEPS', 1)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, accumulation_steps=accum_steps,
            mixup_fn=mixup_cutmix if use_mixup else None
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 每轮结束后释放显存，防止累积导致卡死
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新学习率
        if scheduler:
            if config.SCHEDULER == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # TensorBoard 记录
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Time/epoch', epoch_time, epoch)

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        mode_prefix = "[FINETUNE]" if is_finetune else "[TRAIN]"
        print(f"{mode_prefix} Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s",
              flush=True)

        # 保存最佳模型（按验证准确率，只在有提升时保存）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            # 保存模型
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names,
                'model_name': config.MODEL_NAME,
                'train_mode': config.TRAIN_MODE,
            }, best_model_path)
            print(f"   [SAVE] 保存最佳模型 (验证准确率: {val_acc:.4f})", flush=True)
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n[STOP] 早停: 验证损失连续 {config.EARLY_STOP_PATIENCE} 轮未下降")
            break

    total_time = time.time() - start_time
    history['total_time'] = f"{total_time/60:.1f}分钟"

    print("\n" + "=" * 60, flush=True)
    print(f"[OK] {'微调' if is_finetune else '训练'}完成! 总耗时: {total_time/60:.1f}分钟", flush=True)
    print(f"   最佳验证准确率: {max(history['val_acc']):.4f}", flush=True)
    print(f"   最佳轮次: {history['val_acc'].index(max(history['val_acc'])) + 1}", flush=True)
    print("=" * 60, flush=True)

    # 保存最后一轮模型
    last_model_path = save_dir / "last_model.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'class_names': class_names,
        'model_name': config.MODEL_NAME,
        'train_mode': config.TRAIN_MODE,
    }, last_model_path)

    # 绘制训练曲线
    plot_path = log_dir / "training_curves.png"
    plot_training_history(history, plot_path)

    # 保存训练信息
    save_training_info(history, class_names, save_dir)

    # 在测试集上评估
    print("\n[INFO] 在测试集上评估...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"   测试集准确率: {test_acc:.4f}")
    print(f"   测试集损失: {test_loss:.4f}")

    # TensorBoard 记录测试结果
    if writer is not None:
        writer.add_scalar('Test/loss', test_loss, 0)
        writer.add_scalar('Test/acc', test_acc, 0)
        writer.close()
        print(f"\n[INFO] TensorBoard 日志已保存")
        print(f"   启动命令: tensorboard --logdir={tb_dir}")


if __name__ == "__main__":
    main()
