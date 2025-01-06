import cv2
import os
from ultralytics import YOLO

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Replace with your appropriate YOLO model

# Function to perform YOLO detection on the frame and return the annotated frame
def detect_and_display(frame, image_counter):
    # Perform detection on the frame
    results = model(frame)

    # Render results and show the updated frame
    annotated_frame = results[0].plot()  # This will draw bounding boxes and labels on the frame

    # Save the annotated frame as an image
    result_filename = os.path.join(image_dir, f"image{image_counter}_result.jpg")
    cv2.imwrite(result_filename, annotated_frame)
    print(f"Result image saved: {result_filename}")

    return annotated_frame