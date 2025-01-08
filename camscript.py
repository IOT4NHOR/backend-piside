# backend-piside/camscript.py
import cv2
import time
import os
import json
import requests
from datetime import datetime
from yolo_detection import detect_and_display

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Open the camera
cap = cv2.VideoCapture(0)
time.sleep(2)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize a counter for the image number
image_counter = 1

# Function to save the image
def save_image(frame, image_counter, prefix=''):
    # Generate the filename based on the image number
    filename = os.path.join(image_dir, f"{prefix}image{image_counter}.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")

# Function to select ROIs manually
def select_rois(frame):
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    print("Roi selected")
    cv2.destroyWindow("Select ROIs")
    return rois

# Function to save ROIs to a file
def save_rois(rois, filename='rois.json'):
    with open(filename, 'w') as f:
        json.dump(rois.tolist(), f)
    print(f"ROIs saved to {filename}")

# Function to load ROIs from a file
def load_rois(filename='rois.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            rois = json.load(f)
        print(f"ROIs loaded from {filename}")
        return rois
    else:
        return None

# Function to send the result
def send_result(roi_results):
    url = 'http://iot4gler-iotsmartcam.scnd.space:3000/create'
    for result in roi_results:
        # Here you can add the code to send the result, e.g., via an API call
        if requests.post(url, json=result):
            print("data sent")
        else:
            print("data not sent")
        print(f"Result sent: {result}")

# Capture a frame to select ROIs
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Load existing ROIs or select new ones
rois = load_rois()
if rois is None or input("Use existing ROIs? (y/n): ").strip().lower() != 'y':
    rois = select_rois(frame)
    save_rois(rois)

# Loop to capture images
try:
    start_time = time.time()
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Check if 10 seconds have passed
        if time.time() - start_time >= 3:
            # Save the current frame before processing
            save_image(frame, image_counter, prefix='current_')

            # Draw the selected ROIs on the frame
            for roi in rois:
                x, y, w, h = roi
                if w > 0 and h > 0:  # Ensure the ROI is valid
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    print(f"Invalid ROI: {roi}")

            # Perform YOLO detection and display results for each ROI
            roi_results = []
            all_confidences = []
            for idx, roi in enumerate(rois):
                x, y, w, h = roi
                if w > 0 and h > 0:  # Ensure the ROI is valid
                    roi_frame = frame[y:y+h, x:x+w]
                    annotated_frame, amount_of_person_detected, confidences, class_names = detect_and_display(roi_frame, image_counter)
                    frame[y:y+h, x:x+w] = annotated_frame  # Place the annotated ROI back into the frame
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    roi_results.append({
                        "date_time": current_time,
                        "people": amount_of_person_detected,
                        "object": class_names
                    })
                    all_confidences.extend(confidences)
                else:
                    print(f"Invalid ROI: {roi}")

            # Check for minimum and maximum confidence scores
            if all_confidences:
                min_confidence = min(all_confidences)
                max_confidence = max(all_confidences)
                print(f"Min confidence: {min_confidence}, Max confidence: {max_confidence}")

            # Show the annotated frame in the same window
            cv2.imshow("YOLO Detection", frame)

            # Save the annotated image
            save_image(frame, image_counter)

            # Send the result
            send_result(roi_results)

            # Update the counter and reset it to 1 after 6
            image_counter = (image_counter % 6) + 1

            # Reset the start time
            start_time = time.time()

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Script interrupted by user.")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()