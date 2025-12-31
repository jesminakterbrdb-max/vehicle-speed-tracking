import cv2
import numpy as np
from ultralytics import YOLO

from bytetrack_tracker import ByteTracker
from utils.speed_estimation import SpeedEstimator
from utils.line_crossing import LineCounter
from utils.calibration import Calibration

# -------------------------
# Video Path (Corrected)
# -------------------------
video_path = "traffic.mp4"  # use relative path; file must be in project folder
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# -------------------------
# Calibration (EDIT THIS)
# -------------------------
# Measure pixels between two known road points
cal = Calibration(real_distance_meters=10, pixel_distance=200)
meters_per_pixel = cal.get_factor()

# -------------------------
# Setup Modules
# -------------------------
tracker = ByteTracker()
speed = SpeedEstimator(meters_per_pixel, fps)
counter = LineCounter(y_line=400)

model = YOLO("yolov8n.pt")  # ensure yolov8n.pt is in the project folder

# Vehicle classes (YOLO COCO classes: 2=car,3=motorbike,5=bus,7=truck)
VEHICLES = [2, 3, 5, 7]

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, conf=0.4, iou=0.5, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls)
        if cls in VEHICLES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append([x1, y1, x2, y2])

    # Update tracker
    tracks = tracker.update(detections)

    # Draw counting line
    cv2.line(frame, (0, 400), (1920, 400), (255, 255, 0), 2)

    for tr in tracks:
        x1, y1, x2, y2 = tr.box
        track_id = tr.id
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # SPEED
        speed_kmph = speed.update(track_id, cx, cy)

        # COUNTING
        if counter.check(track_id, cy):
            count += 1

        # OVERSPEED ALERT
        alert = ""
        if speed_kmph > 60:
            alert = "OVERSPEED!"

        # Draw bounding box + speed + ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id} {speed_kmph:.1f} km/h {alert}",
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255) if alert else (255, 255, 255), 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicles Passed: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # Show frame
    cv2.imshow("Vehicle Analytics - ByteTrack", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
