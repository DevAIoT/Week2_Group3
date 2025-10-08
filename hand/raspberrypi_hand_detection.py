"""
Hand Detection with MediaPipe and Servo Control
Detects if hand is open or closed using MediaPipe Hands
Open hand = Toggle door state (open/close)
Closed hand = Idle state
"""

import cv2
import mediapipe as mp
import time

# Try to import RPi.GPIO for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("GPIO library loaded successfully")
except ImportError:
    print("Warning: RPi.GPIO not available. Servo control disabled.")
    GPIO_AVAILABLE = False

def get_capture(src: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera capture and check if available."""
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {src}. Try a different index (0, 1, 2, etc).")
    return cap

import math

class ServoController:
    def __init__(self, pin=18, frequency=50):
        """Initialize servo controller"""
        self.pin = pin
        self.frequency = frequency
        self.pwm = None
        self.door_open = False  # Track door state
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pin, self.frequency)
            self.pwm.start(0)
            print(f"Servo controller initialized on pin {self.pin}")
        else:
            print("Servo controller initialized in simulation mode")
    
    def set_angle(self, angle):
        """Set servo angle (0-180 degrees)"""
        if GPIO_AVAILABLE and self.pwm:
            # Convert angle to duty cycle (2.5% to 12.5% for 0-180 degrees)
            duty_cycle = 2.5 + (angle / 180.0) * 10.0
            self.pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)  # Give servo time to move
            self.pwm.ChangeDutyCycle(0)  # Stop sending signal
        else:
            print(f"Servo simulation: Setting angle to {angle} degrees")
    
    def open_door(self):
        """Move servo to open position"""
        self.set_angle(90)  # Adjust angle as needed for your door mechanism
        self.door_open = True
        print("Door OPENED")
    
    def close_door(self):
        """Move servo to close position"""
        self.set_angle(0)   # Adjust angle as needed for your door mechanism
        self.door_open = False
        print("Door CLOSED")
    
    def toggle_door(self):
        """Toggle door state"""
        if self.door_open:
            self.close_door()
        else:
            self.open_door()
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if GPIO_AVAILABLE:
            if self.pwm:
                self.pwm.stop()
            GPIO.cleanup()
            print("GPIO cleanup completed")

class HandDetector:
    def __init__(self, servo_controller):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Servo controller
        self.servo = servo_controller
        
        # Toggle state management
        self.last_hand_state = False
        self.hand_open_triggered = False
        self.last_trigger_time = 0
        self.debounce_delay = 1.0  # 1 second delay between triggers
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_hand_open(self, hand_landmarks):
        """
        Determine if hand is open or closed
        Returns True if hand is open, False if closed
        """
        # Get key landmarks
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get palm landmarks for reference
        thumb_mcp = hand_landmarks.landmark[2]
        index_mcp = hand_landmarks.landmark[5]  # Index finger base
        middle_mcp = hand_landmarks.landmark[9]  # Middle finger base
        ring_mcp = hand_landmarks.landmark[13]  # Ring finger base
        pinky_mcp = hand_landmarks.landmark[17]  # Pinky finger base
        
        # Calculate distances from fingertips to palm
        index_distance = self.calculate_distance(index_tip, index_mcp)
        middle_distance = self.calculate_distance(middle_tip, middle_mcp)
        ring_distance = self.calculate_distance(ring_tip, ring_mcp)
        pinky_distance = self.calculate_distance(pinky_tip, pinky_mcp)
        thumb_distance = self.calculate_distance(thumb_tip,thumb_mcp)
        
        # Threshold for considering fingers extended
        threshold = 0.1
        
        # Count how many fingers are extended
        fingers_extended = 0
        if index_distance > threshold:
            fingers_extended += 1
        if middle_distance > threshold:
            fingers_extended += 1
        if ring_distance > threshold:
            fingers_extended += 1
        if pinky_distance > threshold:
            fingers_extended += 1
        if thumb_distance > threshold:
            fingers_extended += 1
            
        # Hand is considered open if at least 3 fingers are extended
        return fingers_extended == 5
    
    def detect_hand(self, frame):
        """
        Process frame and detect hand state with servo control
        Returns: (frame_with_drawings, hand_state)
        hand_state: True = open, False = closed or no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        hand_state = False  # Default: False (no hand or closed)
        status_text = "Hand: NOT DETECTED"
        status_color = (0, 0, 255)  # Red
        servo_status = f"Door: {'OPEN' if self.servo.door_open else 'CLOSED'}"
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
                )
                # Check if hand is open or closed
                hand_state = self.is_hand_open(hand_landmarks)
                
        # Servo control logic with debouncing
        current_time = time.time()
        
        if hand_state and not self.last_hand_state:
            # Hand just opened - trigger door toggle if enough time has passed
            if current_time - self.last_trigger_time > self.debounce_delay:
                self.servo.toggle_door()
                self.last_trigger_time = current_time
                self.hand_open_triggered = True
        
        # Update status based on hand state and door state
        if hand_state:
            status_text = "Hand: OPEN (Trigger Active)"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "Hand: CLOSED (Idle)"
            status_color = (0, 165, 255)  # Orange
        
        # Update last hand state
        self.last_hand_state = hand_state
        
        # Display status on frame
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"State: {hand_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(frame, servo_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow
        
        return frame, hand_state
    
    def release(self):
        """Release resources"""
        self.hands.close()


def main():
    """Main function to run hand detection with servo control"""
    # Camera settings
    camera_index = 0  # Use 0 for Raspberry Pi default camera
    width = 320      # Lower resolution for better performance
    height = 240
    
    # Initialize servo controller
    servo = ServoController(pin=18)  # GPIO pin 18 for servo
    
    # Initialize webcam with robust function
    try:
        cap = get_capture(camera_index, width, height)
    except RuntimeError as e:
        print(e)
        servo.cleanup()
        return
    
    # Initialize hand detector with servo controller
    detector = HandDetector(servo)
    
    print("Hand Detection with Servo Control Started!")
    print("- Open your hand: Toggle door state (open/close)")
    print("- Close your hand: Idle state")
    print("- Press 'q' to quit")
    print("- Servo connected to GPIO pin 18")
    print("-" * 50)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand and control servo
            frame, hand_state = detector.detect_hand(frame)
            
            # Print state to console
            door_state = "OPEN" if servo.door_open else "CLOSED"
            if hand_state:
                print(f"Hand: OPEN | Door: {door_state} | Trigger: READY", end='\r')
            else:
                print(f"Hand: CLOSED | Door: {door_state} | Trigger: IDLE", end='\r')
            
            # Display the frame
            cv2.imshow('Hand Detection & Servo Control', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        servo.cleanup()
        print("\nHand Detection and Servo Control Stopped!")


if __name__ == "__main__":
    main()
