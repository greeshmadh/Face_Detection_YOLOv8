import os
import cv2
from ultralytics import YOLO

# Define input and output paths
VIDEO_PATH = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\video\16.mp4'  # Update this path as needed
OUTPUT_PATH = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\output\video\output_video.mp4'

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load the YOLO model
model_path = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\runs\detect\train3\weights\best.pt'
model = YOLO(model_path)  # Load a custom model

# Threshold for object detection
threshold = 0.25

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Read first frame to get video properties
ret, frame = cap.read()
if not ret:
    print(f"Error: Could not read the first frame of {VIDEO_PATH}")
    cap.release()
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

frame_count = 0

while ret:
    frame_count += 1
    if frame is None:
        print(f"Error: Frame {frame_count} is None in {VIDEO_PATH}")
        break

    # Perform object detection
    results = model(frame)[0]

    # Process the results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
print(f"Processed {frame_count} frames in {VIDEO_PATH}")
print(f"Processed video saved to: {OUTPUT_PATH}")

cv2.destroyAllWindows()
