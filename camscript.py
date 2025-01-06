import cv2
import time
import os
import subprocess

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize a counter for the image number
image_counter = 1

# Function to run YOLO detection on the saved image
def run_yolo_detection(image_path):
    subprocess.Popen(['python', 'yolo_detection.py', image_path])

# Function to save the image
def save_image(frame, image_counter):
    # Generate the filename based on the image number
    filename = os.path.join(image_dir, f"image{image_counter}.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")

    # Run YOLO detection on the saved image
    run_yolo_detection(filename)

# Loop to capture images
try:
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Save the image and perform YOLO detection
        save_image(frame, image_counter)

        # Update the counter and reset it to 1 after 6
        image_counter = (image_counter % 6) + 1

        # Wait for 10 seconds before capturing the next photo
        time.sleep(10)

except KeyboardInterrupt:
    print("Script interrupted by user.")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
