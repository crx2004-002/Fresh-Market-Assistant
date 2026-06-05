"""
启动 TensorBoard
"""
import os

# TensorBoard 可能安装在多个位置
possible_paths = [
    r"C:\Users\C\AppData\Roaming\Python\Python312\Scripts\tensorboard.exe",
    r"E:\anaconda\envs\yolov\Scripts\tensorboard.exe",
    r"E:\anaconda\Scripts\tensorboard.exe",
]

# 找到第一个存在的路径
tb_path = None
for p in possible_paths:
    if os.path.exists(p):
        tb_path = p
        break

logdir = r"e:\progect\Fresh Market Assistant\logs\tensorboard"

if tb_path:
    print(f"TensorBoard 路径: {tb_path}")
    print(f"日志目录: {logdir}")
    print()
    print("请在浏览器中打开: http://localhost:6006")
    print("按 Ctrl+C 停止")
    print()
    os.system(f'"{tb_path}" --logdir={logdir}')
else:
    print("找不到 tensorboard.exe")
    print()
    print("请手动运行以下命令:")
    print(f'  "{possible_paths[0]}" --logdir={logdir}')
