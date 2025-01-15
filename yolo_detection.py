# backend-piside/yolo_detection.py
import cv2
import os
from ultralytics import YOLO


# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load the pre-trained YOLO model
model = YOLO('yolo11s.pt')  # Replace with your appropriate YOLO model

# Mapping of class indices to class names for the COCO dataset
class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}


# Function to perform YOLO detection on the frame and return the annotated frame, amount of person detected, confidence scores, and class names
def detect_and_display(frame, result_imagenumber):
    # Perform detection on the frame
    results = model(frame)

    # Render results and show the updated frame

    annotated_frame = results[0].plot()  # This will draw bounding boxes and labels on the frame

    # Save the annotated frame as an image
    result_filename = os.path.join(image_dir, f"image{result_imagenumber}_result.jpg")
    result_imagenumber =  (result_imagenumber % 3) + 1
    cv2.imwrite(result_filename, annotated_frame)
    print(f"Result image saved: {result_filename}")

    # Count the number of people detected with a confidence threshold of 0.6 and collect their confidence scores
    person_confidences = [box.conf for box in results[0].boxes if box.cls == 0 and box.conf >= 0.6]
    amount_of_person_detected = len(person_confidences)

    # Collect class names with a confidence threshold of 0.55, excluding 'person'
    detected_classes = [class_names[int(box.cls)] for box in results[0].boxes if int(box.cls) != 0 and box.conf >= 0.55]

    return annotated_frame, amount_of_person_detected, person_confidences, detected_classes