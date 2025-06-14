from ultralytics import YOLO
from roboflow import Roboflow

# 初始化 Roboflow
rf = Roboflow(api_key="qxXvlNarIRl13bYsGauo")
project = rf.workspace("fyp-3bwmg").project("reloc2-den7l")
version = project.version(1)
dataset = version.download("yolov8")

# 使用 YOLOv8x-pose 模型（專門做 keypoint 任務）
model = YOLO("yolov8x-pose.pt")  # ✅ YOLOv8 的 keypoint 精度最高模型

# 訓練模型
model.train(
    data=f"{dataset.location}/data.yaml",  # Roboflow 提供的 data.yaml 路徑
    epochs=300,        # 關鍵點任務建議訓練時間較長
    imgsz=1280,        # 高解析度可幫助辨識球場線條交點
    batch=8,           # 如果 GPU 有空間可以調高
    name="court_keypoints_yolov8x"
)

# 產出模型權重
import shutil
shutil.copy("runs/pose/court_keypoints_yolov8x/weights/best.pt", "training_notebook/court_keypoint_best.pt")
print("✅ 已輸出 court_keypoint_best.pt")