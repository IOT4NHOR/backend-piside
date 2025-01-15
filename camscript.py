import os
import cv2
import time
import json
import requests
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from yolo_detection import detect_and_display
import numpy as np  # Needed for decoding image data

identity_number = 100  # Initialize identity number

# Create the image directory if it doesn't exist
image_dir = 'image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# URL to fetch images from localhost
image_url = "http://172.16.4.243:2222/current_image"

# Initialize a counter for the image number
image_counter = 1

# Function to save the image
def save_image(frame, image_counter, prefix=''):
    filename = os.path.join(image_dir, f"{prefix}image{image_counter}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Image saved: {filename}")

# Function to fetch an image from the localhost server
def fetch_image():
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            # Decode the image data
            image_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
            else:
                print("Error: Failed to decode the image.")
        else:
            print(f"Failed to fetch image: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
    return None

# Function to select ROIs manually
def select_rois(frame):
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    print("ROIs selected")
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

# Function to get the latest image file from the image folder
def get_latest_image():
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(image_dir, x)), reverse=True)
    return os.path.join(image_dir, files[0])

# Function to send the result
def send_result(roi_results, identity_number, last_update_time):
    url = 'http://iot4gler-iotsmartcam.scnd.space:3000/create'
    payload = {
        "identity_number": identity_number,
        "results": roi_results,
        "last_update_time": last_update_time
    }
    try:
        print(f"Sending result: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print(f"Failed to send data: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# HTTP request handler class
class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/latest_image':
            latest_image = get_latest_image()
            if latest_image is None:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No image found")
                return

            with open(latest_image, 'rb') as f:
                image_data = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(image_data)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

# Function to start the HTTP server
def start_server():
    server_address = ('', 2223)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Starting server on port 2223...")
    httpd.serve_forever()

# Fetch an initial frame for ROI selection
frame = fetch_image()
if frame is None:
    print("Error: Failed to fetch the initial image.")
    exit()

# Load existing ROIs or select new ones
rois = load_rois()
if rois is None or input("Use existing ROIs? (y/n): ").strip().lower() != 'y':
    rois = select_rois(frame)
    save_rois(rois)

# Start the HTTP server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

# Loop to process images
try:
    start_time = time.time()
    while True:
        # Check if 10 seconds have passed
        if time.time() - start_time >= 1:
            frame = fetch_image()
            if frame is None:
                print("Error: Failed to fetch the image.")
                break

            # Save the current frame before processing
            save_image(frame, image_counter, prefix='current_')

            # Draw the selected ROIs on the frame
            for idx, roi in enumerate(rois):
                x, y, w, h = roi
                if w > 0 and h > 0:  # Ensure the ROI is valid
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"ROI {idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Perform YOLO detection and display results for each ROI
            roi_results = []
            all_confidences = []
            last_update_time = None
            for idx, roi in enumerate(rois):
                x, y, w, h = roi
                if w > 0 and h > 0:  # Ensure the ROI is valid
                    roi_frame = frame[y:y+h, x:x+w]
                    annotated_frame, amount_of_person_detected, confidences, class_names = detect_and_display(roi_frame, image_counter)
                    frame[y:y+h, x:x+w] = annotated_frame  # Place the annotated ROI back into the frame
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    roi_results.append({
                        "roi_number": idx + 1,
                        "people": amount_of_person_detected,
                        "object": class_names
                    })
                    all_confidences.extend(confidences)
                    last_update_time = current_time
                else:
                    print(f"Invalid ROI: {roi}")

            # Check for minimum and maximum confidence scores
            if all_confidences:
                min_confidence = min(all_confidences)
                max_confidence = max(all_confidences)
                print(f"Min confidence: {min_confidence}, Max confidence: {max_confidence}")

            # Show the annotated frame
            cv2.imshow("YOLO Detection", frame)

            # Save the annotated image
            save_image(frame, image_counter)

            # Send the result
            # send_result(roi_results, identity_number, last_update_time)

            # Update the counter and reset it to 1 after 6
            image_counter = (image_counter % 60) + 1

            # Reset the start time
            start_time = time.time()

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Script interrupted by user.")

# Close OpenCV windows
cv2.destroyAllWindows()