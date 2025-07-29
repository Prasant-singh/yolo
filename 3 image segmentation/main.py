from ultralytics import YOLO
import cv2

model= YOLO("yolov8s-seg.pt")

cap=cv2.VideoCapture("2 yolo croping/doghum.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame)
    annotated_frame = results[0].plot()
    class_names = results[0].names
    cv2.imshow("YOLOv8 Segmentation", annotated_frame)
    for result in results:
        for box in result.boxes:
            if box.conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = box.cls[0]
                class_name = class_names[int(cls)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Annotated Cropped Frame", annotated_frame[y1:y2, x1:x2])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()