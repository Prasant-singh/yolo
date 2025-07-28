from ultralytics import YOLO     

model = YOLO("runs/detect/yolo_custom_model/weights/best.pt")
results = model.val(data="data.yaml", imgsz=640)