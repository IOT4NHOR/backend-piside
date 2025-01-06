import cv2
import time
import os
from yolo_detection import detect_and_display

more2 = 1

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

# Function to save the image
def save_image(frame, image_counter):
    # Generate the filename based on the image number
    filename = os.path.join(image_dir, f"image{image_counter}.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")

# Function to select ROIs manually
def select_rois(frame):
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROIs")
    return rois

# Capture a frame to select ROIs
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Select ROIs manually once
rois = select_rois(frame)
print(f"Selected ROIs: {rois}")

# Loop to capture images
try:
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform YOLO detection and display results for each ROI
        for roi in rois:
            x, y, w, h = roi
            if w > 0 and h > 0:  # Ensure the ROI is valid
                roi_frame = frame[y:y+h, x:x+w]
                annotated_frame = detect_and_display(roi_frame, image_counter)
                frame[y:y+h, x:x+w] = annotated_frame  # Place the annotated ROI back into the frame
            else:
                print(f"Invalid ROI: {roi}")

        # Show the annotated frame in the same window
        cv2.imshow("YOLO Detection", frame)

        # Save the annotated image
        save_image(frame, image_counter)

        # Update the counter and reset it to 1 after 6
        image_counter = (image_counter % 6) + 1

        # Wait for 10 seconds before capturing the next photo
        time.sleep(10)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Script interrupted by user.")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()