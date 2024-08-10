import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\runs\detect\train3\weights\best.pt")  # Update this path if needed

# Open a connection to the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process video frames in real-time
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
