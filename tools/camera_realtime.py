"""
水果识别模型 - 摄像头实时识别脚本
使用摄像头实时检测并识别水果
添加结果稳定机制，避免结果跳动
支持轻量 TTA（实时模式下 TTA 数量减少以保持帧率）
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

import config
from tools.model_factory import create_model, get_val_transforms, get_tta_transforms
from class_names_cn import get_cn_name

# 实时模式下 TTA 数量（减少以保持帧率）
REALTIME_TTA_NUM = 3


class ResultStabilizer:
    """结果稳定器 - 检测到水果后锁定显示"""

    def __init__(self, lock_duration=1.0):
        """
        Args:
            lock_duration: 锁定时长（秒），默认1秒
        """
        self.lock_duration = lock_duration
        self.locked_result = None
        self.lock_start_time = 0

    def update(self, predictions):
        """
        更新预测结果
        - 锁定期内：始终返回锁定的结果
        - 锁定结束后：用新预测更新锁定结果
        """
        now = time.time()

        # 锁定期内，直接返回锁定结果
        if self.locked_result is not None and (now - self.lock_start_time) < self.lock_duration:
            return self.locked_result

        # 锁定结束，更新锁定结果
        if predictions:
            self.locked_result = predictions
            self.lock_start_time = now

        return predictions

    def reset(self):
        self.locked_result = None
        self.lock_start_time = 0
        self.locked_result = None
        self.lock_counter = 0


class FruitRealtimeDetector:
    """水果实时检测器"""

    def __init__(self, model_path=None, use_tta=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = f"{config.SAVE_DIR}/best_model.pth"

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        self.model_name = checkpoint.get('model_name', config.MODEL_NAME)

        # 使用统一模型工厂创建模型
        self.model = create_model(self.model_name, self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据变换
        self.transform = get_val_transforms(config.IMAGE_SIZE)

        # TTA 变换（实时模式下使用少量 TTA）
        self.use_tta = use_tta
        if use_tta:
            self.tta_transforms = get_tta_transforms(config.IMAGE_SIZE, REALTIME_TTA_NUM)

        self.stabilizer = ResultStabilizer(lock_duration=1.0)

        # 过滤阈值配置（平衡准确率和召回率）
        self.CONFIDENCE_THRESHOLD = 0.60   # 最低置信度
        self.MARGIN_THRESHOLD = 0.20       # Top1与Top2最小差距
        self.ENTROPY_THRESHOLD = 0.65      # 信息熵上限

        # 调试模式
        self.debug_mode = False

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # 预加载中文字体（只加载一次）
        self.cn_font = None
        self.cn_font_small = None
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
            try:
                self.cn_font = ImageFont.truetype(fp, 22)
                self.cn_font_small = ImageFont.truetype(fp, 17)
                break
            except (IOError, OSError):
                continue
        if self.cn_font is None:
            self.cn_font = ImageFont.load_default()
            self.cn_font_small = self.cn_font

        print(f"[OK] 模型加载成功")
        print(f"   模型: {self.model_name}")
        print(f"   设备: {self.device}")
        print(f"   TTA: {'启用' if use_tta else '关闭'}")
        print(f"   按 'q' 退出, 's' 截图, 'r' 重置, 'd' 调试模式")

    def predict(self, pil_image, top_k=3):
        """
        预测（支持 TTA）
        Args:
            pil_image: PIL.Image 对象
            top_k: 返回前 K 个结果
        Returns:
            predictions: 预测结果列表
            entropy: 归一化信息熵
        """
        if self.use_tta:
            # TTA 模式：对多个变换取平均
            tta_probs = []
            with torch.no_grad():
                for transform in self.tta_transforms:
                    input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    tta_probs.append(probs)
            probabilities = torch.stack(tta_probs).mean(dim=0).squeeze(0)
        else:
            # 标准模式
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, top_k)

        # 计算信息熵（衡量模型的"犹豫程度"）
        all_probs = probabilities.cpu().numpy()
        entropy = -np.sum(all_probs * np.log(all_probs + 1e-10))
        max_entropy = np.log(self.num_classes)  # 均匀分布时的最大熵
        normalized_entropy = entropy / max_entropy     # 0=完全确定, 1=完全随机

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({'class': self.class_names[idx.item()], 'confidence': prob.item()})

        return predictions, normalized_entropy

    def draw_predictions(self, frame, predictions, fps, entropy=0.0):
        """在帧上绘制预测结果（整帧只做一次 OpenCV->PIL 转换）"""
        height, width = frame.shape[:2]

        # OpenCV BGR -> PIL RGB（只转换一次）
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 半透明背景（在PIL上绘制）
        overlay = Image.new('RGB', (370, 170), (0, 0, 0))
        pil_img.paste(overlay, (10, 10), None)

        # 标题
        draw.text((20, 12), "水果识别 (每秒更新)", font=self.cn_font, fill=(0, 255, 255))

        # 预测结果（三重过滤防止幻觉）
        top1_conf = predictions[0]['confidence'] if predictions else 0
        top2_conf = predictions[1]['confidence'] if len(predictions) > 1 else 0
        margin = top1_conf - top2_conf

        # 三重判断：置信度够高 + 差距够大 + 熵够低
        is_valid = (top1_conf >= self.CONFIDENCE_THRESHOLD and
                    margin >= self.MARGIN_THRESHOLD and
                    entropy < self.ENTROPY_THRESHOLD)

        if is_valid:
            for i, pred in enumerate(predictions[:3]):
                y_pos = 50 + i * 32
                cn_name = get_cn_name(pred['class'])
                confidence = pred['confidence'] * 100

                if confidence > 85:
                    color = (0, 255, 0)
                elif confidence > 75:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                text = f"{i+1}. {cn_name}: {confidence:.1f}%"
                draw.text((20, y_pos), text, font=self.cn_font, fill=color)
        else:
            draw.text((20, 50), "未检测到水果", font=self.cn_font, fill=(150, 150, 150))

        # PIL RGB -> OpenCV BGR（只转换一次）
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # FPS和帮助信息用OpenCV绘制（纯英文，不需要中文支持）
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Q: Quit | S: Screenshot | R: Reset", (20, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        return frame

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERR] 无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n🎥 摄像头已打开")

        screenshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERR] 无法读取视频帧")
                break

            # 转换为 PIL 图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            raw_predictions, entropy = self.predict(pil_image, top_k=3)

            # 稳定器始终工作
            predictions = self.stabilizer.update(raw_predictions)

            # 调试模式输出
            if self.debug_mode:
                print(f"\n[DEBUG] Frame {self.frame_count}")
                print(f"  Entropy: {entropy:.3f}")
                for i, pred in enumerate(raw_predictions[:3]):
                    print(f"  {i+1}. {pred['class']}: {pred['confidence']*100:.1f}%")

            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time

            frame = self.draw_predictions(frame, predictions, self.fps, entropy)

            cv2.imshow('Fruit Recognition - Press Q to Quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                cv2.imwrite(f"screenshot_{screenshot_count}.jpg", frame)
                print(f"[SAVE] 截图已保存: screenshot_{screenshot_count}.jpg")
            elif key == ord('r'):
                self.stabilizer.reset()
                print("[RESET] 已重置")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"[DEBUG] 调试模式: {'开启' if self.debug_mode else '关闭'}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] 总帧数: {self.frame_count}, 平均FPS: {self.fps:.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="水果实时识别")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID")
    parser.add_argument("--model", type=str, default=None, help="模型文件路径")
    parser.add_argument("--tta", action="store_true", help="启用轻量 TTA (3个变换)")
    args = parser.parse_args()

    print("=" * 50)
    print("🍓 水果实时识别系统")
    if args.tta:
        print("   模式: 轻量 TTA")
    print("=" * 50)

    try:
        detector = FruitRealtimeDetector(model_path=args.model, use_tta=args.tta)
        detector.run(camera_id=args.camera)
    except FileNotFoundError as e:
        print(f"[ERR] 错误: {e}")
    except Exception as e:
        print(f"[ERR] 错误: {e}")
        raise


if __name__ == "__main__":
    main()
