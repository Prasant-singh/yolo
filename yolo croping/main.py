import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap= cv2.VideoCapture("doghum.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, stream=True)
    if not results:
        continue

    for result in results:
        classes= result.names
        for box in result.boxes:
            if box.conf[0] > 0.85:
                x1, y1, x2, y2 = box.xyxy[0].int()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = classes[cls]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cropped_image = frame[y1:y2, x1:x2]
                center= (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                if cropped_image.size > 0.5:
                    cv2.imwrite(f"cropped_{class_name}.jpg", cropped_image)



        cv2.imshow("YOLOv8 Tracking", frame)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()