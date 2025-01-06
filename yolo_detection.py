import cv2
import os
import numpy as np
from ultralytics import YOLO

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Replace with your appropriate YOLO model

# Function to perform Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    print(f"indices: {indices}")  # Debug print
    print(f"boxes: {boxes}")  # Debug print
    if isinstance(indices, np.ndarray) and len(indices) > 0:
        return [boxes[i] for i in indices.flatten()]
    elif isinstance(indices, list) and len(indices) > 0:
        return [boxes[i] for i in indices]
    else:
        return []

# Function to perform YOLO detection on the frame and return the annotated frame
def detect_and_display(frame, image_counter):
    # Perform detection on the frame
    results = model(frame)

    # Extract bounding boxes and scores
    boxes = []
    scores = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates from the first element
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(float(box.conf))

    # Apply Non-Maximum Suppression
    nms_boxes = non_max_suppression(boxes, scores, iou_threshold=0.4)

    # Draw bounding boxes on the frame
    for box in nms_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the annotated frame as an image
    result_filename = os.path.join(image_dir, f"image{image_counter}_result.jpg")
    cv2.imwrite(result_filename, frame)
    print(f"Result image saved: {result_filename}")

    return frame