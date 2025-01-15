import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Global variable for the current frame
current_frame = None

# Lock for thread safety
frame_lock = threading.Lock()

def capture_camera():
    global current_frame
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Lock and update the current frame
        with frame_lock:
            current_frame = frame.copy()

    # Release resources
    cap.release()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame

        if self.path == '/current_image':
            with frame_lock:
                if current_frame is None:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"No frame available")
                    return

                # Encode the frame to JPEG
                _, buffer = cv2.imencode('.jpg', current_frame)
                jpg_as_text = buffer.tobytes()

            # Send HTTP response with the image
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(jpg_as_text)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

def start_server():
    server_address = ('', 2222)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Starting server on port 2222...")
    httpd.serve_forever()

if __name__ == "__main__":
    # Start the camera capture in a separate thread
    camera_thread = threading.Thread(target=capture_camera)
    camera_thread.daemon = True
    camera_thread.start()

    # Start the HTTP server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    # Display frames in the main thread
    cv2.startWindowThread()

    while True:
        with frame_lock:
            if current_frame is not None:
                cv2.imshow('frame', current_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup OpenCV resources
    cv2.destroyAllWindows()
