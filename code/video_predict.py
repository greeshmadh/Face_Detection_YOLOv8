# import os
# import cv2
# from ultralytics import YOLO

# VIDEOS_DIR = os.path.join('.', 'videos')

# video_path = r'C:\Users\Nishita Gopi\Documents\video\input_vid1.mp4'
# video_path_out = '{}_out.mp4'.format(video_path)

# # Open video file
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print(f"Error: Could not open video file {video_path}")
#     exit()

# # Read first frame to get video properties
# ret, frame = cap.read()
# if not ret:
#     print("Error: Could not read the first frame of the video")
#     cap.release()
#     exit()

# H, W, _ = frame.shape

import os
import cv2
from ultralytics import YOLO

# Define input and output directories
VIDEOS_DIR = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\video'
OUTPUT_DIR = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\output\video'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the YOLO model
model_path = r'C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\runs\detect\train3\weights\best.pt'
model = YOLO(model_path)  # Load a custom model

# Threshold for object detection
threshold = 0.25

# Process each video in the input directory
for video_file in os.listdir(VIDEOS_DIR):
    video_path = os.path.join(VIDEOS_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    video_out_path = os.path.join(OUTPUT_DIR, f'{video_name}_out.mp4')

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    # Read first frame to get video properties
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of {video_file}")
        cap.release()
        continue

    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    frame_count = 0

    while ret:
        frame_count += 1
        if frame is None:
            print(f"Error: Frame {frame_count} is None in {video_file}")
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
    print(f"Frame count: {frame_count}")
    print(f"Processed video saved to: {video_out_path}")

cv2.destroyAllWindows()
