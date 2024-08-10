import os
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")  # Load an official model
model = YOLO(r"C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\runs\detect\train3\weights\best.pt")  # Uncomment if you want to use your custom model

# Define input and output directories
input_dir = r"C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\dataset\small"  # Change this to your input folder path
output_dir = r"C:\Users\Admin\Documents\GREESHMA\CDSAML\face_detection\output\small"  # Change this to your output folder path

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the input directory
for file_name in os.listdir(input_dir):
    # Get the full file path
    file_path = os.path.join(input_dir, file_name)

    # Predict with the model
    results = model(file_path)

    # Check if results is a list (batch prediction) or a single result
    if isinstance(results, list):
        for i, result in enumerate(results):
            result.save(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_batch{i}.jpg"))
            
    else:
        results.save(os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.jpg"))

print("Processing complete.")