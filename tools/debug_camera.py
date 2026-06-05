"""
摄像头诊断脚本 - 查看模型对每一帧的原始输出
用于调试幻觉问题
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import config
from class_names_cn import get_cn_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint = torch.load(f"{config.SAVE_DIR}/best_model.pth", map_location=device)
    class_names = checkpoint['class_names']

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3), torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, len(class_names))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(int(config.IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    print("=" * 60)
    print("🔍 诊断模式 - 查看模型原始输出")
    print("   对准空白背景，观察输出分布")
    print("   按 'q' 退出")
    print("=" * 60)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]

        # 获取所有类别概率
        top5_probs, top5_indices = torch.topk(probs, 5)

        # 每10帧打印一次详细信息
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"\n--- 帧 {frame_count} ---")
            for prob, idx in zip(top5_probs, top5_indices):
                cn = get_cn_name(class_names[idx.item()])
                print(f"  {cn:8s} ({class_names[idx.item()]:12s}): {prob.item()*100:6.2f}%")

            # 打印所有类别概率的统计信息
            all_probs = probs.cpu().numpy()
            print(f"\n  统计: max={all_probs.max()*100:.2f}%  "
                  f"mean={all_probs.mean()*100:.2f}%  "
                  f"std={all_probs.std()*100:.2f}%  "
                  f"熵={-(all_probs * np.log(all_probs + 1e-10)).sum():.3f}")

        # 在画面上显示Top1
        top1_cn = get_cn_name(class_names[top5_indices[0].item()])
        top1_conf = top5_probs[0].item() * 100
        cv2.putText(frame, f"Top1: {top1_cn} {top1_conf:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Top2: {get_cn_name(class_names[top5_indices[1].item()])} {top5_probs[1].item()*100:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Debug Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
