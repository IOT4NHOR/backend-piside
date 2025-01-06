import cv2
import sys
from ultralytics import YOLO
import os


image_counter = 1

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def save_image(frame, image_counter):
    # Generate the filename based on the image number
    filename = os.path.join(image_dir, f"image{image_counter}_result.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Replace with your appropriate YOLO model (e.g., yolov8n.pt)

# Get the image path from the command-line argument
image_path = sys.argv[1]

# Load the image
frame = cv2.imread(image_path)

# Perform detection
results = model(frame)

# Since results is a list, we can iterate through it and display each result
for result in results:
    # Display the results (objects detected, etc.)
    result.show()  # This automatically shows the image with bounding boxes and labels

    # Optionally, save the result image with detections drawn
    output_image = 'detected_' + image_path
    # Save the image and perform YOLO detection
    save_image(frame, image_counter)

# Update the counter and reset it to 1 after 6
image_counter = (image_counter % 6) + 1

# Wait until a key is pressed to close the displayed window
cv2.waitKey(0)
cv2.destroyAllWindows()





