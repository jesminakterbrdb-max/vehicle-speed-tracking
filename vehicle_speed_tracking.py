# --------------------------------------------
# VEHICLE SPEED TRACKING SYSTEM (FULL SCRIPT)
# --------------------------------------------

from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort  # Make sure sort.py is in same folder

# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("yolov8n.pt")  # Download automatically

# ---------------------------
# Initialize Tracker (SORT)
# ---------------------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ---------------------------
# Video Input
# ---------------------------
video_path = "traffic.mp4"    # <---- CHANGE THIS TO YOUR VIDEO FILE
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------------------
# Calibration (EDIT THIS VALUE)
# ---------------------------
# meters per pixel value must be calibrated for accuracy
meters_per_pixel = 0.05  # temporary value

# Store previous positions of tracked vehicles
vehicle_positions = {}

# ---------------------------
# Main Loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    # VEHICLE CLASSES (from COCO)
    vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Convert YOLO detections to SORT format
    for obj in results.boxes:
        cls = int(obj.cls)
        if cls in vehicle_class_ids:
            x1, y1, x2, y2 = obj.xyxy[0]
            score = float(obj.conf)
            detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections)
    tracked = tracker.update(detections)

    # Draw Tracking + Compute Speed
    for d in tracked:
        x1, y1, x2, y2, track_id = d
        track_id = int(track_id)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Speed calculation
        if track_id in vehicle_positions:
            old_x, old_y = vehicle_positions[track_id]
            pixel_distance = np.sqrt((cx - old_x) ** 2 + (cy - old_y) ** 2)

            meters = pixel_distance * meters_per_pixel
            speed_mps = meters * fps
            speed_kmph = speed_mps * 3.6
        else:
            speed_kmph = 0

        # Update position
        vehicle_positions[track_id] = (cx, cy)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display ID + Speed
        cv2.putText(
            frame,
            f"ID:{track_id}  {speed_kmph:.1f} km/h",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow("Vehicle Speed Tracking", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
