"""
水果识别模型 - 配置文件
针对 RTX 4070 8GB 显存优化
支持首次训练和二次微调

优化策略:
1. 首次训练: 冻结骨干，只训练分类层，学习率较高
2. 二次微调: 渐进式解冻，学习率降低10倍
3. 使用 Label Smoothing 提升泛化能力
4. ConvNeXtV2-Tiny 29M参数，ImageNet-22K预训练，适合39类水果分类
"""

# ==================== 训练模式 ====================
# TRAIN_MODE: "first" = 首次训练, "finetune" = 二次微调
TRAIN_MODE = "first"

# ==================== 数据集配置 ====================
# 数据集根目录（按类别子文件夹组织）
DATA_DIR = "./dataset"

# 图片大小（ConvNeXtV2-Tiny 推荐 288）
IMAGE_SIZE = 288

# 训练集/验证集/测试集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== 训练配置 ====================
# 预训练模型选择
# ConvNeXtV2-Tiny: ~29M参数，2023年SOTA CNN，ImageNet-22K预训练
MODEL_NAME = "convnextv2_tiny"

# 批次大小（RTX 4070 8G 显存，ConvNeXtV2-Tiny 288px 用 32）
BATCH_SIZE = 32

# 训练轮数（配合早停机制，设大一点没关系）
NUM_EPOCHS = 50

# 学习率（首次训练，冻结骨干时用较高学习率）
LEARNING_RATE = 0.001

# 学习率调度器: "cosine", "step", "plateau"
SCHEDULER = "cosine" #余弦退火

# 早停轮数（验证集loss连续多少轮不下降就停止）
EARLY_STOP_PATIENCE = 10

# 是否冻结预训练模型的参数（只训练最后的分类层）
FREEZE_BACKBONE = True

# ==================== 二次微调配置 ====================
# 微调起始模型路径
FINETUNE_MODEL_PATH = "./checkpoints/best_model.pth"

# 微调时的学习率（比首次训练小10倍）
FINETUNE_LEARNING_RATE = 0.0001

# 微调时的训练轮数
FINETUNE_NUM_EPOCHS = 30

# 渐进式解冻级别
# 0: 只解冻分类层
# 1: 解冻最后的block + 分类层
# -1: 全部解冻
FINETUNE_UNFREEZE_LEVEL = 1

# ==================== 类别权重 ====================
# 使用逆频率类别权重，提升少数类的学习信号
USE_CLASS_WEIGHTS = True

# ==================== 系统配置 ====================
# 数据加载线程数（Windows建议2，避免多进程问题）
NUM_WORKERS = 2
USE_GPU = True
RANDOM_SEED = 42
SAVE_DIR = "./checkpoints"
LOG_DIR = "./logs"

# ==================== 训练优化 ====================
# AMP 混合精度训练（RTX 4070 Tensor Core 加速，节省40%显存）
USE_AMP = True

# Label Smoothing（标签平滑，防止过拟合）
LABEL_SMOOTHING = 0.1

# 优化器 weight_decay（AdamW 正则化强度）
WEIGHT_DECAY = 0.01

# 数据增强强度（"light" 或 "heavy" 或 "trivial"）
# "light" = 保留水果颜色特征的温和增强，适合细粒度分类
AUGMENTATION_MODE = "light"

# Mixup / CutMix 数据增强
# 水果分类不适合Mixup（破坏空间结构），已关闭
USE_MIXUP = False
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.4
MIXUP_PROB = 0.4

# ==================== TTA 测试时增强 ====================
TTA_ENABLED = False
TTA_NUM_AUGS = 10

# ==================== 多模型集成 ====================
ENSEMBLE_MODEL_PATHS = []
ENSEMBLE_WEIGHTS = [0.4, 0.35, 0.25]
