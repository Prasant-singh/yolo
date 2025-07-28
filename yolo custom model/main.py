from ultralytics import YOLO

# # Load a pre-trained model or initialize from scratch
model = YOLO("yolov8n.pt")

#from ultralytics import YOLO

# # Load a pre-trained model or initialize from scratch
model = YOLO("yolov8n.pt")

# # Train the model
results = model.train(
data="data.yaml",
epochs=150,
imgsz=640,
batch=16,
name="yolo_custom_model",
verbose=False               
)

