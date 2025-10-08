"""
Combined Hand Detection and Arduino Control
Detects hand open/closed state using MediaPipe and sends toggle commands to Arduino
- Open hand: Sends toggle command to Arduino (alternates between True/False)
- Closed hand: No action (idle state)
- Each hand opening triggers a toggle
"""
import os
import threading
import cv2
import mediapipe as mp
import math
import time
from datetime import datetime
from flask import Flask, Response

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_W = int(os.environ.get("FRAME_W", "320"))
FRAME_H = int(os.environ.get("FRAME_H", "240"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", "20"))
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "/dev/ttyACM0")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "0") == "1"

CURRENT_STATE = "0"

# Try to import serial for Arduino communication
try:
    import serial
    SERIAL_AVAILABLE = True
    print("Serial library loaded successfully")
except ImportError:
    print("Warning: pyserial not available. Arduino control disabled.")
    print("Install with: pip install pyserial")
    SERIAL_AVAILABLE = False

def get_capture(src: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera capture and check if available."""
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {src}. Try a different index (0, 1, 2, etc).")
    return cap

class ArduinoController:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, timeout=1):
        """Initialize Arduino controller for Raspberry Pi"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        
    def connect(self):
        """Establish serial connection with Arduino"""
        if not SERIAL_AVAILABLE:
            print("Serial not available - running in simulation mode")
            self.connected = True
            return True
            
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Give Arduino time to reset after connection
            print(f"✓ Connected to Arduino on {self.port}")
            self.connected = True
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Arduino: {e}")
            print("Available ports to try: /dev/ttyUSB0, /dev/ttyACM0")
            self.connected = False
            return False
    
    def send_command(self, state):
        """Send True/False command to Arduino"""
        if not self.connected:
            print("Arduino not connected")
            return
            
        command = "1"

        if SERIAL_AVAILABLE and self.ser and self.ser.is_open:
            command = "1" if state else "0"
            self.ser.write(command.encode())
            print(f"→ Arduino: {state} ({command}) - Servo to {180 if state else 0}°")
            
            # Try to read response
            try:
                time.sleep(0.1)
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode().strip()
                    if response:
                        print(f"← Arduino: {response}")
            except:
                pass
        else:
            # Simulation mode
            command = "1" if state else "0"
            print(f"[SIM] → Arduino: {state} ({command}) - Servo to {180 if state else 0}°")

        CURRENT_STATE = command
    
    def disconnect(self):
        """Close serial connection"""
        if SERIAL_AVAILABLE and self.ser and self.ser.is_open:
            self.ser.close()
        print("Arduino disconnected")

class HandDetector:
    def __init__(self, arduino_controller):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Arduino controller
        self.arduino = arduino_controller
        
        # Toggle state management
        self.last_hand_state = False
        self.current_toggle_state = False  # Current Arduino state
        self.last_trigger_time = 0
        self.debounce_delay = 1.0  # 1 second delay between triggers
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_hand_open(self, hand_landmarks):
        """
        Determine if hand is open or closed
        Returns True if ALL 5 fingers are extended
        """
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get palm landmarks for reference
        thumb_mcp = hand_landmarks.landmark[2]
        index_mcp = hand_landmarks.landmark[5]
        middle_mcp = hand_landmarks.landmark[9]
        ring_mcp = hand_landmarks.landmark[13]
        pinky_mcp = hand_landmarks.landmark[17]
        
        # Calculate distances from fingertips to palm
        thumb_distance = self.calculate_distance(thumb_tip, thumb_mcp)
        index_distance = self.calculate_distance(index_tip, index_mcp)
        middle_distance = self.calculate_distance(middle_tip, middle_mcp)
        ring_distance = self.calculate_distance(ring_tip, ring_mcp)
        pinky_distance = self.calculate_distance(pinky_tip, pinky_mcp)
        
        # Threshold for considering fingers extended
        threshold = 0.1
        
        # Count how many fingers are extended
        fingers_extended = 0
        if thumb_distance > threshold:
            fingers_extended += 1
        if index_distance > threshold:
            fingers_extended += 1
        if middle_distance > threshold:
            fingers_extended += 1
        if ring_distance > threshold:
            fingers_extended += 1
        if pinky_distance > threshold:
            fingers_extended += 1
            
        # Hand is considered open if all 5 fingers are extended
        return fingers_extended == 5
    
    def detect_hand_and_control_arduino(self, frame):
        """
        Process frame, detect hand state, and control Arduino with toggle logic
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_state = False
        status_text = "Hand: NOT DETECTED"
        status_color = (0, 0, 255)  # Red
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
                )
                # Check if hand is open
                hand_state = self.is_hand_open(hand_landmarks)
        
        # Arduino control logic with toggle
        current_time = time.time()
        
        if hand_state and not self.last_hand_state:
            # Hand just opened - trigger toggle if enough time has passed
            if current_time - self.last_trigger_time > self.debounce_delay:
                self.current_toggle_state = not self.current_toggle_state
                self.arduino.send_command(self.current_toggle_state)
                self.last_trigger_time = current_time
                print(f"🖐️  Hand opened - Toggle to: {self.current_toggle_state}")
        
        # Update status based on hand state
        if hand_state:
            status_text = "Hand: OPEN (Trigger Ready)"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "Hand: CLOSED (Idle)"
            status_color = (0, 165, 255)  # Orange
        
        # Update last hand state
        self.last_hand_state = hand_state
        
        # Display status on frame
        arduino_status = f"Arduino: {'ON' if self.current_toggle_state else 'OFF'} (180°/0°)"
        
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Hand State: {hand_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, arduino_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow
        
        return frame, hand_state
    
    def release(self):
        """Release MediaPipe resources"""
        self.hands.close()
# -----------------------------
# MJPEG stream (Flask in thread)
# -----------------------------
app = Flask(__name__)
_latest_jpg = None
_stream_lock = threading.Lock()

def _mjpeg_generator():
    # Stream boundary must match content-type header boundary
    while True:
        with _stream_lock:
            data = _latest_jpg
        if data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(1.0 / max(TARGET_FPS, 1))

@app.route("/stream")
def stream():
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    return Response(f"Door State: {'ON' if CURRENT_STATE == '1' else 'OFF'}\n",
                    mimetype='text/plain')

def start_stream_server():
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True),
        daemon=True
    )
    t.start()
    print("📡 MJPEG stream available at http://<pi-ip>:5000/stream")

# -----------------------------
# Main loop
# -----------------------------
def main():
    print("Hand Detection + Arduino Control + MJPEG Stream")
    print("=" * 54)

    # Arduino
    arduino = ArduinoController()
    if not arduino.connect():
        print("Warning: continuing without Arduino connection")

    # Camera
    try:
        cap = get_capture(CAMERA_INDEX, FRAME_W, FRAME_H)
        print(f"✓ Camera initialized: {FRAME_W}x{FRAME_H}")
    except RuntimeError as e:
        print(f"✗ Camera error: {e}")
        return

    # Detector
    detector = HandDetector(arduino_controller=arduino)

    # Start MJPEG server
    start_stream_server()

    print("\nSystem ready!")
    print("- Open your hand: toggles Arduino state (True ↔ False)")
    print("- Closed hand: idle")
    print("- Stream at: http://<pi-ip>:5000/stream")
    if SHOW_WINDOW:
        print("- Local preview window enabled (SHOW_WINDOW=1)")
    print("- Press Ctrl+C to quit")
    print("-" * 54)

    # Loop & publish frames
    frame_interval = 1.0 / max(TARGET_FPS, 1)
    try:
        last_time = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera")
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            # Process
            frame, hand_state = detector.detect_hand_and_control_arduino(frame)

            # Encode once per loop (publish to stream)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with _stream_lock:
                    global _latest_jpg
                    _latest_jpg = buf.tobytes()

            # Optional local preview
            if SHOW_WINDOW:
                cv2.imshow("Hand Detection + Arduino Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Simple fps cap
            now = time.time()
            sleep_for = frame_interval - (now - last_time)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_time = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        detector.release()
        arduino.disconnect()
        print("System shutdown complete!")

if __name__ == "__main__":
    main()