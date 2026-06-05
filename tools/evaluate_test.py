"""
水果识别模型 - 测试集评估脚本
在测试集上进行全面评估，生成详细报告
支持单模型评估、TTA 评估、多模型集成评估
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import config
from tools.model_factory import (
    create_model, load_checkpoint_model, get_val_transforms, get_tta_transforms
)
from tools.class_names_cn import FRUIT_NAMES_CN


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path=None):
        """初始化评估器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        if model_path is None:
            model_path = f"{config.SAVE_DIR}/best_model.pth"

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        self.train_mode = checkpoint.get('train_mode', 'first')
        self.best_val_acc = checkpoint.get('val_acc', 'N/A')
        self.model_name = checkpoint.get('model_name', config.MODEL_NAME)

        # 创建模型（使用统一模型工厂）
        self.model = create_model(self.model_name, self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据预处理
        self.transform = get_val_transforms(config.IMAGE_SIZE)

        # TTA 变换
        self.tta_transforms = get_tta_transforms(config.IMAGE_SIZE, config.TTA_NUM_AUGS)

        # 类别中文名称 (从 class_names_cn.py 导入完整的39类映射)
        self.class_names_cn = FRUIT_NAMES_CN

        print(f"✅ 模型加载成功")
        print(f"   模型: {self.model_path}")
        print(f"   模型类型: {self.model_name}")
        print(f"   训练模式: {self.train_mode}")
        print(f"   设备: {self.device}")

    def load_test_dataset(self):
        """加载测试集"""
        test_dir = Path(config.DATA_DIR) / "test"

        if not test_dir.exists():
            raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

        test_dataset = datasets.ImageFolder(test_dir, transform=self.transform)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=False
        )

        print(f"\n📊 测试集信息:")
        print(f"   总样本数: {len(test_dataset)}")
        print(f"   类别数: {len(test_dataset.classes)}")

        return test_loader

    def evaluate(self, test_loader):
        """在测试集上评估"""
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        return all_predictions, all_labels, all_probs

    def evaluate_with_tta(self, test_dir):
        """
        使用 TTA 在测试集上评估
        对每张图片应用多个变换，取预测平均值
        """
        test_dataset = datasets.ImageFolder(test_dir, transform=None)  # 不应用变换
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        all_predictions = []
        all_labels = []
        all_probs = []

        print(f"\n🔍 TTA 评估中 (每张图片 {len(self.tta_transforms)} 个变换)...")

        with torch.no_grad():
            for i, (img_pil, label) in enumerate(test_loader):
                img_pil = img_pil[0]  # DataLoader 返回的是 batch
                label = label[0]

                # 对每个变换进行预测
                tta_probs = []
                for transform in self.tta_transforms:
                    img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
                    output = self.model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    tta_probs.append(probs)

                # 取平均
                avg_probs = torch.stack(tta_probs).mean(dim=0)
                _, pred = torch.max(avg_probs, 1)

                all_predictions.append(pred.cpu().numpy()[0])
                all_labels.append(label.numpy())
                all_probs.append(avg_probs.cpu().numpy()[0])

                if (i + 1) % 50 == 0:
                    print(f"   已处理: {i + 1}/{len(test_dataset)}")

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        return all_predictions, all_labels, all_probs

    def calculate_metrics(self, predictions, labels, probs):
        """计算各种评估指标"""
        # 总体准确率
        accuracy = np.mean(predictions == labels)

        # 每个类别的准确率
        class_accuracies = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                class_accuracies[class_name] = {
                    'accuracy': class_acc,
                    'count': int(np.sum(class_mask)),
                    'correct': int(np.sum((predictions[class_mask] == labels[class_mask])))
                }

        # 混淆矩阵
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true, pred in zip(labels, predictions):
            confusion_matrix[true][pred] += 1

        # 精确率、召回率、F1值
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for i in range(self.num_classes):
            tp = confusion_matrix[i][i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)

        # 平均指标
        avg_precision = np.mean(precision_per_class)
        avg_recall = np.mean(recall_per_class)
        avg_f1 = np.mean(f1_per_class)

        # 置信度统计
        max_probs = np.max(probs, axis=1)
        avg_confidence = np.mean(max_probs)

        metrics = {
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'class_accuracies': class_accuracies,
            'confusion_matrix': confusion_matrix,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'avg_confidence': avg_confidence,
            'total_samples': len(labels)
        }

        return metrics

    def print_results(self, metrics):
        """打印评估结果"""
        print("\n" + "=" * 70)
        print("📊 测试集评估结果")
        print("=" * 70)

        print(f"\n🎯 总体指标:")
        print(f"   准确率 (Accuracy):    {metrics['accuracy']*100:.2f}%")
        print(f"   平均精确率 (Precision): {metrics['avg_precision']*100:.2f}%")
        print(f"   平均召回率 (Recall):   {metrics['avg_recall']*100:.2f}%")
        print(f"   平均F1值 (F1-Score):   {metrics['avg_f1']*100:.2f}%")
        print(f"   平均置信度:           {metrics['avg_confidence']*100:.2f}%")
        print(f"   测试样本数:           {metrics['total_samples']}")

        print(f"\n📈 各类别详细结果:")
        print("-" * 70)
        print(f"{'类别':<12} {'中文名':<8} {'准确率':<10} {'样本数':<8} {'正确数':<8}")
        print("-" * 70)

        for class_name in self.class_names:
            info = metrics['class_accuracies'][class_name]
            cn_name = self.class_names_cn.get(class_name, class_name)
            print(f"{class_name:<12} {cn_name:<8} {info['accuracy']*100:.2f}%   {info['count']:<8} {info['correct']:<8}")

        print("-" * 70)

        # 找出最难和最简单的类别
        sorted_classes = sorted(metrics['class_accuracies'].items(),
                               key=lambda x: x[1]['accuracy'])

        print(f"\n⚠️  最难识别的类别:")
        for class_name, info in sorted_classes[:3]:
            cn_name = self.class_names_cn.get(class_name, class_name)
            print(f"   {cn_name} ({class_name}): {info['accuracy']*100:.2f}%")

        print(f"\n✅ 最易识别的类别:")
        for class_name, info in sorted_classes[-3:]:
            cn_name = self.class_names_cn.get(class_name, class_name)
            print(f"   {cn_name} ({class_name}): {info['accuracy']*100:.2f}%")

    def plot_confusion_matrix(self, confusion_matrix, save_path):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 使用中文类别名
        cn_names = [self.class_names_cn.get(c, c) for c in self.class_names]

        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('混淆矩阵', fontsize=16)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(self.class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(cn_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(cn_names, fontsize=10)

        # 在矩阵中添加数值
        thresh = confusion_matrix.max() / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if confusion_matrix[i, j] > thresh else "black",
                       fontsize=8)

        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_xlabel('预测标签', fontsize=12)
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 混淆矩阵已保存: {save_path}")

    def plot_class_accuracies(self, class_accuracies, save_path):
        """绘制类别准确率柱状图"""
        fig, ax = plt.subplots(figsize=(14, 6))

        classes = list(class_accuracies.keys())
        accuracies = [class_accuracies[c]['accuracy'] * 100 for c in classes]
        cn_names = [self.class_names_cn.get(c, c) for c in classes]

        # 根据准确率设置颜色
        colors = []
        for acc in accuracies:
            if acc >= 95:
                colors.append('#2ecc71')  # 绿色
            elif acc >= 90:
                colors.append('#f39c12')  # 黄色
            else:
                colors.append('#e74c3c')  # 红色

        bars = ax.bar(range(len(classes)), accuracies, color=colors)

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(cn_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_title('各类别识别准确率', fontsize=16)
        ax.set_ylim(0, 105)
        ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%线')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%线')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 类别准确率图已保存: {save_path}")

    def save_report(self, metrics, save_dir):
        """保存评估报告"""
        # 转换numpy类型为Python原生类型
        def to_python_type(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_python_type(x) for x in obj]
            return obj

        report = {
            "model_path": str(self.model_path),
            "train_mode": self.train_mode,
            "best_val_acc": float(self.best_val_acc) if isinstance(self.best_val_acc, (int, float, np.floating)) else self.best_val_acc,
            "test_accuracy": float(metrics['accuracy']),
            "avg_precision": float(metrics['avg_precision']),
            "avg_recall": float(metrics['avg_recall']),
            "avg_f1": float(metrics['avg_f1']),
            "avg_confidence": float(metrics['avg_confidence']),
            "total_samples": int(metrics['total_samples']),
            "class_results": {
                name: {
                    "cn_name": self.class_names_cn.get(name, name),
                    "accuracy": float(info['accuracy']),
                    "count": int(info['count']),
                    "correct": int(info['correct'])
                }
                for name, info in metrics['class_accuracies'].items()
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        report_path = save_dir / "test_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📋 评估报告已保存: {report_path}")
        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="水果识别模型 - 测试集评估")
    parser.add_argument("--model", type=str, default=None, help="模型文件路径")
    parser.add_argument("--tta", action="store_true", help="启用 TTA 测试时增强")
    parser.add_argument("--ensemble", action="store_true", help="启用多模型集成评估")
    parser.add_argument("--models", nargs='+', help="集成模型路径列表")

    args = parser.parse_args()

    print("=" * 70)
    print("🍓 水果识别模型 - 测试集评估")
    if args.tta:
        print("   模式: TTA 测试时增强")
    elif args.ensemble:
        print("   模式: 多模型集成")
    else:
        print("   模式: 标准评估")
    print("=" * 70)

    try:
        if args.ensemble:
            # 集成评估
            _run_ensemble_evaluation(args)
        else:
            # 单模型评估（可选 TTA）
            _run_single_evaluation(args)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        raise


def _run_single_evaluation(args):
    """单模型评估"""
    evaluator = ModelEvaluator(model_path=args.model)
    test_loader = evaluator.load_test_dataset()

    test_dir = Path(config.DATA_DIR) / "test"

    if args.tta:
        print("\n🔍 正在 TTA 评估...")
        predictions, labels, probs = evaluator.evaluate_with_tta(test_dir)
    else:
        print("\n🔍 正在评估...")
        predictions, labels, probs = evaluator.evaluate(test_loader)

    metrics = evaluator.calculate_metrics(predictions, labels, probs)
    evaluator.print_results(metrics)

    # 保存图表
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        log_dir / "confusion_matrix.png"
    )
    evaluator.plot_class_accuracies(
        metrics['class_accuracies'],
        log_dir / "class_accuracies.png"
    )
    evaluator.save_report(metrics, Path(config.SAVE_DIR))

    print("\n" + "=" * 70)
    print(f"✅ 评估完成!")
    print(f"   测试集准确率: {metrics['accuracy']*100:.2f}%")
    print("=" * 70)


def _run_ensemble_evaluation(args):
    """多模型集成评估"""
    from tools.ensemble_predict import EnsemblePredictor

    model_paths = args.models or getattr(config, 'ENSEMBLE_MODEL_PATHS', [])
    if not model_paths:
        model_paths = [
            "./checkpoints/dinov2_vitb14/best_model.pth",
            "./checkpoints/convnextv2_base/best_model.pth",
            "./checkpoints/efficientvit_l2/best_model.pth",
        ]

    weights = getattr(config, 'ENSEMBLE_WEIGHTS', None)
    predictor = EnsemblePredictor(model_paths, weights)

    test_dir = Path(config.DATA_DIR) / "test"
    test_dataset = datasets.ImageFolder(test_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_predictions = []
    all_labels = []
    all_probs = []

    print(f"\n🔍 集成评估中 ({len(test_dataset)} 张图片)...")

    with torch.no_grad():
        for i, (img_pil, label) in enumerate(test_loader):
            img_pil = img_pil[0]
            label = label[0]

            results = predictor.predict_pil(img_pil, top_k=1, use_tta=True, tta_num=5)
            pred_class = predictor.class_names.index(results[0]['class'])

            all_predictions.append(pred_class)
            all_labels.append(label.numpy())

            if (i + 1) % 50 == 0:
                print(f"   已处理: {i + 1}/{len(test_dataset)}")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_predictions == all_labels)
    print(f"\n✅ 集成评估完成!")
    print(f"   测试集准确率: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
