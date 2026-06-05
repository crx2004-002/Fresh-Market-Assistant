"""
水果识别模型 - 统一模型工厂
支持所有预训练模型的创建、数据变换、TTA
2024-2025 新增: DINOv2, ConvNeXtV2, EfficientViT
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==================== 模型创建 ====================

def create_model(model_name, num_classes, pretrained=True):
    """
    统一模型创建入口
    Args:
        model_name: 模型名称
        num_classes: 分类数
        pretrained: 是否使用预训练权重
    Returns:
        model: PyTorch 模型
    """
    # 传统 torchvision 模型
    if model_name == "resnet50":
        return _create_resnet50(num_classes, pretrained)
    elif model_name == "efficientnet_b0":
        return _create_efficientnet_b0(num_classes, pretrained)
    elif model_name == "mobilenet_v3":
        return _create_mobilenet_v3(num_classes, pretrained)

    # 2024 新模型
    elif model_name == "dinov2_vitb14":
        return _create_dinov2("dinov2_vitb14", num_classes, pretrained)
    elif model_name == "dinov2_vits14":
        return _create_dinov2("dinov2_vits14", num_classes, pretrained)
    elif model_name == "dinov2_vitl14":
        return _create_dinov2("dinov2_vitl14", num_classes, pretrained)
    elif model_name == "convnextv2_base":
        return _create_convnextv2(num_classes, pretrained)
    elif model_name == "convnextv2_tiny":
        return _create_convnextv2_tiny(num_classes, pretrained)
    elif model_name == "efficientvit_l2":
        return _create_efficientvit(num_classes, pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def _create_resnet50(num_classes, pretrained):
    """ResNet50 - 传统 CNN 基线"""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = _make_classifier_head(in_features, num_classes)
    return model


def _create_efficientnet_b0(num_classes, pretrained):
    """EfficientNet-B0 - 轻量级 CNN"""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = _make_classifier_head(in_features, num_classes)
    return model


def _create_mobilenet_v3(num_classes, pretrained):
    """MobileNetV3-Small - 移动端模型"""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def _create_dinov2(variant, num_classes, pretrained):
    """
    DINOv2 - Meta 2024 自监督视觉基础模型
    特点: 通用特征极强，小数据集微调效果远超 ImageNet 预训练

    使用 timm 库加载（从 HuggingFace 下载，国内更稳定）
    timm 模型名映射:
        dinov2_vits14 -> vit_small_patch14_dinov2
        dinov2_vitb14 -> vit_base_patch14_dinov2
        dinov2_vitl14 -> vit_large_patch14_dinov2
    """
    import timm

    # torch.hub 变体名 -> timm 模型名
    timm_name_map = {
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
    }

    timm_name = timm_name_map.get(variant)
    if timm_name is None:
        raise ValueError(f"未知的 DINOv2 变体: {variant}")

    print(f"   通过 timm 加载 {timm_name}...")

    # 本地模型路径（优先使用）
    local_checkpoints = [
        f"./checkpoints/model.safetensors",
        f"./checkpoints/model_small.safetensors",
        f"./checkpoints/{timm_name}.safetensors",
        f"./checkpoints/{timm_name}.pth",
    ]

    try:
        # 先尝试从本地加载
        import os
        local_path = None
        for p in local_checkpoints:
            if os.path.exists(p):
                local_path = p
                break

        if local_path:
            print(f"   从本地加载: {local_path}")
            model = timm.create_model(timm_name, pretrained=False, num_classes=0)
            from safetensors.torch import load_file
            state_dict = load_file(local_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            # 从 HuggingFace 镜像下载
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    except Exception as e:
        raise RuntimeError(
            f"加载 DINOv2 失败: {e}\n"
            f"请确保已安装: pip install timm safetensors\n"
            f"模型文件应放在 ./checkpoints/model.safetensors"
        )

    embed_dim = model.num_features  # vits=384, vitb=768, vitl=1024

    # 组合 backbone + 分类头
    model = nn.Sequential(
        model,
        nn.Dropout(0.3),
        nn.Linear(embed_dim, 512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model


def _create_convnextv2(num_classes, pretrained):
    """
    ConvNeXtV2-Base - 微软 2023 最强 CNN
    特点: 纯 CNN 架构，性能超越大多数 ViT
    """
    import timm
    model = timm.create_model(
        'convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=pretrained,
        num_classes=0  # 移除原始分类头
    )
    in_features = model.num_features  # 1024
    model = nn.Sequential(
        model,
        _make_classifier_head(in_features, num_classes)
    )
    return model


def _create_convnextv2_tiny(num_classes, pretrained):
    """
    ConvNeXtV2-Tiny - 微软 2023 高效 CNN
    特点: ~29M参数，7x7大核深度卷积，LayerNorm，GELU
          ImageNet-22K预训练后微调ImageNet-1K，特征提取能力强
    适合: 中等数据集(10K-100K图片)的细粒度分类
    """
    import timm
    model = timm.create_model(
        'convnextv2_tiny.fcmae_ft_in22k_in1k',
        pretrained=pretrained,
        num_classes=0  # 移除原始分类头
    )
    in_features = model.num_features  # 768
    model = nn.Sequential(
        model,
        _make_classifier_head(in_features, num_classes)
    )
    return model


def _create_efficientvit(num_classes, pretrained):
    """
    EfficientViT-L2 - 微软 2024 高效 Vision Transformer
    特点: 线性复杂度注意力，适合部署
    """
    import timm
    model = timm.create_model(
        'efficientvit_l2.r288_in1k',
        pretrained=pretrained,
        num_classes=0  # 移除原始分类头
    )
    in_features = model.num_features
    model = nn.Sequential(
        model,
        _make_classifier_head(in_features, num_classes)
    )
    return model


def _make_classifier_head(in_features, num_classes):
    """创建标准分类头"""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )


# ==================== 模型信息 ====================

def get_model_info(model_name):
    """获取模型的基本信息"""
    info = {
        "resnet50":         {"params": "25.6M",  "image_size": 224, "source": "torchvision"},
        "efficientnet_b0":  {"params": "5.3M",   "image_size": 224, "source": "torchvision"},
        "mobilenet_v3":     {"params": "2.5M",   "image_size": 224, "source": "torchvision"},
        "dinov2_vits14":    {"params": "22M",    "image_size": 518, "source": "torch.hub"},
        "dinov2_vitb14":    {"params": "86M",    "image_size": 518, "source": "torch.hub"},
        "dinov2_vitl14":    {"params": "300M",   "image_size": 518, "source": "torch.hub"},
        "convnextv2_base":  {"params": "89M",    "image_size": 288, "source": "timm"},
        "convnextv2_tiny":  {"params": "29M",    "image_size": 288, "source": "timm"},
        "efficientvit_l2":  {"params": "120M",   "image_size": 288, "source": "timm"},
    }
    return info.get(model_name, {"params": "unknown", "image_size": 224, "source": "unknown"})


def get_default_image_size(model_name):
    """获取模型的默认输入尺寸"""
    return get_model_info(model_name)["image_size"]


# ==================== 数据变换 ====================

def get_train_transforms(image_size=224, mode="trivial"):
    """
    训练集数据增强
    Args:
        image_size: 目标图像尺寸（int 或 tuple）
        mode: "light" = 轻量增强, "heavy" = 手动重增强, "trivial" = TrivialAugmentWide
    """
    # 兼容新版 torchvision：确保 size 为 tuple
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size

    if mode == "trivial":
        # TrivialAugmentWide: Google 2021 零参数增强策略
        # 随机选1种变换施加随机强度，是小数据集(2-5K)的最优选择
        # 保留 RandomResizedCrop + HFlip + Normalize（TrivialAugment不包含这些）
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    elif mode == "heavy":
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.5,
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:  # light
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


def get_val_transforms(image_size=224):
    """验证/测试集变换（确定性）"""
    # 兼容新版 torchvision：确保 size 为 tuple
    if isinstance(image_size, int):
        size = (image_size, image_size)
        resize_size = int(image_size * 1.14)
    else:
        size = image_size
        resize_size = int(max(image_size) * 1.14)

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def get_tta_transforms(image_size=224, num_augments=10):
    """
    TTA 测试时增强变换列表
    包含: 原图、翻转、旋转、5-crop、颜色抖动等
    """
    # 兼容新版 torchvision：确保 size 为 tuple
    if isinstance(image_size, int):
        size = (image_size, image_size)
        resize_size = int(image_size * 1.14)
        resize_size_25 = int(image_size * 1.25)
    else:
        size = image_size
        resize_size = int(max(image_size) * 1.14)
        resize_size_25 = int(max(image_size) * 1.25)

    tta_list = [
        # 1. 原图（标准变换）
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 2. 水平翻转
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 3. 垂直翻转
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 4. 旋转 +10°
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 5. 旋转 -10°
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 6. 左上角裁剪
        transforms.Compose([
            transforms.Resize(resize_size_25),
            transforms.FiveCrop(size),
            transforms.Lambda(lambda crops: crops[0]),  # 左上
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 7. 右上角裁剪
        transforms.Compose([
            transforms.Resize(resize_size_25),
            transforms.FiveCrop(size),
            transforms.Lambda(lambda crops: crops[1]),  # 右上
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 8. 左下角裁剪
        transforms.Compose([
            transforms.Resize(resize_size_25),
            transforms.FiveCrop(size),
            transforms.Lambda(lambda crops: crops[2]),  # 左下
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 9. 右下角裁剪
        transforms.Compose([
            transforms.Resize(resize_size_25),
            transforms.FiveCrop(size),
            transforms.Lambda(lambda crops: crops[3]),  # 右下
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # 10. 颜色抖动
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    ]
    return tta_list[:num_augments]


# ==================== 检查点加载 ====================

def load_checkpoint_model(checkpoint_path, device='cpu'):
    """
    从检查点加载模型
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    Returns:
        model: 加载好权重的模型
        class_names: 类别名称列表
        model_name: 模型名称
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)
    model_name = checkpoint.get('model_name', 'resnet50')

    # 创建模型
    model = create_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, model_name


def freeze_backbone(model, model_name):
    """
    冻结骨干网络，只训练分类头
    Args:
        model: 模型
        model_name: 模型名称
    """
    if model_name.startswith("dinov2"):
        # DINOv2: 冻结 backbone (index 0)，只训练分类头 (index 1+)
        for name, param in model.named_parameters():
            if not name.startswith("1.") and not name.startswith("2.") and not name.startswith("3.") and not name.startswith("4."):
                param.requires_grad = False
        print("   [OK] 已冻结 DINOv2 backbone，只训练分类头")
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("   [OK] 已冻结 ResNet50 backbone，只训练分类层")
    elif model_name == "efficientnet_b0":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("   [OK] 已冻结 EfficientNet backbone，只训练分类层")
    elif model_name == "mobilenet_v3":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("   [OK] 已冻结 MobileNet backbone，只训练分类层")
    elif model_name.startswith("convnextv2"):
        for name, param in model.named_parameters():
            if "1" not in name:  # nn.Sequential 的第二个元素是分类头
                param.requires_grad = False
        print(f"   [OK] 已冻结 {model_name} backbone，只训练分类层")
    elif model_name == "efficientvit_l2":
        for name, param in model.named_parameters():
            if "1" not in name:
                param.requires_grad = False
        print("   [OK] 已冻结 EfficientViT backbone，只训练分类层")
    else:
        # 通用：冻结除 classifier/fc 之外的所有层
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name and "head" not in name:
                param.requires_grad = False
        print("   [OK] 已冻结 backbone，只训练分类层")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable:,} / {total:,}")


def apply_gradual_unfreezing(model, model_name, unfreeze_level):
    """
    渐进式解冻策略
    Args:
        model: 模型
        model_name: 模型名称
        unfreeze_level: 解冻级别
            0: 只解冻分类层
            1: 解冻最后一层 + 分类层
            2: 解冻最后两层 + 分类层
            -1: 全部解冻
    """
    if unfreeze_level == -1:
        for param in model.parameters():
            param.requires_grad = True
        print("   [OK] 已解冻所有层")
    elif unfreeze_level == 0:
        freeze_backbone(model, model_name)
    else:
        # 对于 DINOv2 和新模型，level > 0 时全部解冻（全量微调）
        for param in model.parameters():
            param.requires_grad = True
        print(f"   [OK] 已解冻所有层 (unfreeze_level={unfreeze_level})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   可训练参数: {trainable:,}")
    return model
