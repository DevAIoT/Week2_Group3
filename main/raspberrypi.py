"""
Combined Hand Detection, Voice Commands, and Arduino Control
Detects hand open/closed state using MediaPipe and voice commands using Vosk, 
sends toggle commands to Arduino
- Open hand: Sends toggle command to Arduino (alternates between True/False)
- Closed hand: No action (idle state)
- Voice commands: "on", "off", "turn on", "turn off"
- Each hand opening or voice command triggers Arduino control
"""
import os
import threading
import cv2
import mediapipe as mp
import math
import time
import queue
import json
import sys
import numpy as np
from datetime import datetime
from flask import Flask, Response

# Voice recognition imports (with fallback)
try:
    import vosk
    import sounddevice as sd
    VOSK_AVAILABLE = True
    print("Vosk voice recognition loaded successfully")
except ImportError:
    print("Warning: Vosk not available. Voice commands disabled.")
    print("Install with: pip install vosk sounddevice")
    VOSK_AVAILABLE = False

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_W = int(os.environ.get("FRAME_W", "320"))
FRAME_H = int(os.environ.get("FRAME_H", "240"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", "20"))
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "/dev/ttyACM0")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "0") == "1"

# Voice recognition configuration
VOICE_SAMPLE_RATE = 16000
VOICE_MODEL_PATH = os.environ.get("VOICE_MODEL_PATH", "vosk-model-small-en-us-0.15")
VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "1") == "1"

# Mic level SSE rate
UPDATE_HZ = float(os.environ.get("AUDIO_HZ", "20"))  # times per second

class DoorState:
    def __init__(self):
        self._lock = threading.Lock()
        self._unlocked = False

    def set(self, unlocked: bool):
        with self._lock:
            self._unlocked = bool(unlocked)

    def get(self) -> bool:
        with self._lock:
            return self._unlocked

DOOR_STATE = DoorState()

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
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False

    def connect(self):
        if not SERIAL_AVAILABLE:
            print("Serial not available - running in simulation mode")
            self.connected = True
            return True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"✓ Connected to Arduino on {self.port}")
            self.connected = True
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Arduino: {e}")
            print("Available ports to try: /dev/ttyUSB0, /dev/ttyACM0")
            self.connected = False
            return False

    # === DOOR / SERVO ===
    def send_servo(self, unlocked: bool):
        """Control the door servo (hand gestures). Updates DOOR_STATE."""
        if not self.connected:
            print("Arduino not connected (servo)")
            return
        cmd = "1" if unlocked else "0"  # your existing firmware mapping for servo
        if (not SERIAL_AVAILABLE) or (self.ser is None) or (not self.ser.is_open):
            print(f"[SIM] → SERVO: {'UNLOCK' if unlocked else 'LOCK'} ({cmd})")
            DOOR_STATE.set(unlocked)
            return
        try:
            self.ser.write((cmd + "\n").encode())
            print(f"→ SERVO: {'UNLOCK' if unlocked else 'LOCK'} ({cmd})")
            time.sleep(0.05)
            if self.ser.in_waiting > 0:
                resp = self.ser.readline().decode(errors="ignore").strip()
                if resp:
                    print(f"← Arduino: {resp}")
            DOOR_STATE.set(unlocked)
        except Exception as e:
            print("Serial write failed (servo):", e)

    # === LIGHTS (VOICE) ===
    def send_lights(self, on: bool):
        """Control lights via voice commands only (no door state changes)."""
        if not self.connected:
            print("Arduino not connected (lights)")
            return
        cmd = "turn_on" if on else "turn_off"
        if (not SERIAL_AVAILABLE) or (self.ser is None) or (not self.ser.is_open):
            print(f"[SIM] → LIGHTS: {cmd}")
            return
        try:
            self.ser.write((cmd + "\n").encode())
            print(f"→ LIGHTS: {cmd}")
            time.sleep(0.05)
            if self.ser.in_waiting > 0:
                resp = self.ser.readline().decode(errors="ignore").strip()
                if resp:
                    print(f"← Arduino: {resp}")
        except Exception as e:
            print("Serial write failed (lights):", e)

    def disconnect(self):
        if SERIAL_AVAILABLE and self.ser and self.ser.is_open:
            self.ser.close()
        print("Arduino disconnected")

def _to_dbfs(rms: float) -> float:
    rms = max(rms, 1e-12)
    return 20.0 * math.log10(rms)

_level_lock = threading.Lock()
_last_level = {"rms": 0.0, "peak": 0.0, "dbfs": -120.0, "level": 0.0}
class VoiceController:
    def __init__(self, arduino_controller):
        """Initialize voice recognition controller + shared mic level metrics."""
        self.arduino = arduino_controller
        self.running = False
        self.audio_queue: "queue.Queue[bytes|None]" = queue.Queue()
        self.recognizer = None
        self.voice_thread: threading.Thread | None = None
        self.audio_stream = None

        self.THROTTLE_PARTIAL_SEC = 0.5  # print partials at most twice per second
        self.THROTTLE_FINAL_SEC   = 0.3  # minimal gap between finals (safety)
        self.MIN_FINAL_WORDS      = 1    # ignore super-short finals (bump to 2/3 if you want)
        self.MIN_FINAL_CHARS      = 3

        self._last_partial_text = ""
        self._last_partial_ts   = 0.0
        self._last_final_text   = ""
        self._last_final_ts     = 0.0

        if VOSK_AVAILABLE and VOICE_ENABLED:
            self._initialize_vosk()

    def start(self):
        """Start voice recognition + volume meter."""
        if not (VOSK_AVAILABLE and VOICE_ENABLED and self.recognizer):
            print("Voice recognition not available (missing Vosk/model or VOICE_ENABLED=0)")
            return False
        try:
            self.running = True
            self.audio_stream = sd.RawInputStream(
                samplerate=VOICE_SAMPLE_RATE,
                blocksize=8000,       # adjust for latency vs CPU
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            self.voice_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.voice_thread.start()
            print("🎤 Voice recognition started — say 'on' / 'off' / 'turn on' / 'turn off'")
            return True
        except Exception as e:
            print(f"Error starting voice recognition: {e}")
            self.running = False
            try:
                if self.audio_stream:
                    self.audio_stream.stop()
                    self.audio_stream.close()
            except Exception:
                pass
            return False

    def _initialize_vosk(self):
        """Initialize Vosk model and recognizer."""
        try:
            if not os.path.exists(VOICE_MODEL_PATH):
                print(f"Vosk model not found at {VOICE_MODEL_PATH} — voice disabled.")
                return False
            model = vosk.Model(VOICE_MODEL_PATH)
            grammar = ["turn on", "turn off"]
            self.recognizer = vosk.KaldiRecognizer(model, VOICE_SAMPLE_RATE, grammar)
            self.recognizer.SetWords(True)
            print(f"Voice recognition initialized with model: {VOICE_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        # queue raw bytes for STT
        self.audio_queue.put(bytes(indata))
        # compute level metrics here (no extra stream)
        try:
            x = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
            if x.size:
                peak = float(np.max(np.abs(x)))
                rms = float(np.sqrt(np.mean(x**2)))
                dbfs = float(_to_dbfs(rms))
                level = float(min(1.0, max(0.0, rms * 3.0)))  # adjust gain factor as needed
                with _level_lock:
                    _last_level.update({"rms": rms, "peak": peak, "dbfs": dbfs, "level": level})
        except Exception:
            pass

    def _process_audio(self):
        """Consume mic buffers, run Vosk, print throttled partials/finals, and trigger Arduino commands."""
        LOUDNESS_THRESHOLD = 0.015
        
        while self.running:
            try:
                data = self.audio_queue.get(timeout=1.0)
                if data is None:
                    break

                # Check current loudness
                with _level_lock:
                    loud_enough = _last_level["rms"] > LOUDNESS_THRESHOLD

                if not loud_enough:
                    # too quiet — skip processing this buffer
                    continue
                
                now = time.time()

                if self.recognizer and self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = (result.get("text") or "").strip().lower()

                    # Throttle finals + ignore dupes/short noise
                    if text:
                        if (now - self._last_final_ts) >= self.THROTTLE_FINAL_SEC and (
                            text != self._last_final_text
                            and (len(text) >= self.MIN_FINAL_CHARS)
                            and (len(text.split()) >= self.MIN_FINAL_WORDS)
                        ):
                            print(f"\n🎤 Final recognized: '{text}'")
                            self._last_final_text = text
                            self._last_final_ts = now

                            # Commands
                            if text == "turn on":
                                print(">>> Voice Command: ON <<<")
                                self.arduino.send_lights(True)
                            elif text == "turn off":
                                print(">>> Voice Command: OFF <<<")
                                self.arduino.send_lights(False)

                    # Reset partial state after a final
                    self._last_partial_text = ""
                    self._last_partial_ts = now

                else:
                    # Throttled partials (only when changed AND enough time passed)
                    if self.recognizer:
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = (partial.get("partial") or "").strip()
                        if partial_text and partial_text != self._last_partial_text:
                            if (now - self._last_partial_ts) >= self.THROTTLE_PARTIAL_SEC:
                                print(f"🎙️ Listening: {partial_text}...", end="\r", flush=True)
                                self._last_partial_text = partial_text
                                self._last_partial_ts = now

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice processing error: {e}")

    def stop(self):
        """Stop voice recognition cleanly."""
        self.running = False
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                pass
        if self.voice_thread:
            self.audio_queue.put(None)  # unblock
            self.voice_thread.join(timeout=2.0)
        print("🎤 Voice recognition stopped")

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
                self.arduino.send_servo(self.current_toggle_state)
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
    unlocked = DOOR_STATE.get()
    return Response(
        f'{{"unlocked": {str(unlocked).lower()}, "state": "{ "UNLOCKED" if unlocked else "LOCKED" }"}}\n',
        mimetype="application/json"
    )

@app.route("/audio/level")
def audio_level():
    def gen():
        period = 1.0 / max(UPDATE_HZ, 1)
        while True:
            with _level_lock:
                lvl = dict(_last_level)
            yield f"data: {json.dumps(lvl)}\n\n"
            time.sleep(period)
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    }
    return Response(gen(), headers=headers)

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
    print("Hand Detection + Voice Commands + Arduino Control + MJPEG Stream")
    print("=" * 64)

    # Arduino
    arduino = ArduinoController()
    if not arduino.connect():
        print("Warning: continuing without Arduino connection")

    # Voice Controller
    voice_controller = VoiceController(arduino)
    voice_started = voice_controller.start()

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
    if voice_started:
        print("- Voice cmds: 'on' / 'off' / 'turn on' / 'turn off'")
    print("- Video:   http://<pi-ip>:5000/stream")
    print("- Mic SSE: http://<pi-ip>:5000/audio/level")
    print("- Status:  http://<pi-ip>:5000/status")
    if SHOW_WINDOW:
        print("- Local preview window enabled (SHOW_WINDOW=1)")
    print("- Ctrl+C to quit")
    print("-" * 60)

    frame_interval = 1.0 / max(TARGET_FPS, 1)
    try:
        last_time = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break
            frame = cv2.flip(frame, 1)
            frame, _ = detector.detect_hand_and_control_arduino(frame)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with _stream_lock:
                    global _latest_jpg
                    _latest_jpg = buf.tobytes()
            if SHOW_WINDOW:
                cv2.imshow("Hand + Voice + Arduino", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            now = time.time()
            sleep_for = frame_interval - (now - last_time)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_time = time.time()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        detector.release()
        voice_controller.stop()
        arduino.disconnect()
        print("Shutdown complete")

if __name__ == "__main__":
    main()