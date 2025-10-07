"""
Hand Detection with MediaPipe
Detects if hand is open or closed using MediaPipe Hands
Open hand = True (door opens)
Closed hand = False (door closes)
"""

import cv2
import mediapipe as mp

def get_capture(src: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera capture and check if available."""
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {src}. Try a different index (0, 1, 2, etc).")
    return cap
import math

class HandDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
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
        Process frame and detect hand state
        Returns: (frame_with_drawings, hand_state)
        hand_state: True = open, False = closed or no hand detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        hand_state = False  # Default: False (no hand or closed)
        status_text = "Hand: NOT DETECTED"
        status_color = (0, 0, 255)  # Red
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Check if hand is open or closed
                hand_state = self.is_hand_open(hand_landmarks)
                
                if hand_state:
                    status_text = "Hand: OPEN (Door Opens)"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "Hand: CLOSED (Door Closes)"
                    status_color = (0, 165, 255)  # Orange
        
        # Display status on frame
        cv2.putText(frame, status_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3) 
        
        
        
        # Display boolean value
        cv2.putText(frame, f"State: {hand_state}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        return frame, hand_state
    
    def release(self):
        """Release resources"""
        self.hands.close()


def main():
    """Main function to run hand detection"""
    # Camera settings
    camera_index = 1  # Use 0 for Raspberry Pi default camera
    width = 1280
    height = 720
    # Initialize webcam with robust function
    try:
        cap = get_capture(camera_index, width, height)
    except RuntimeError as e:
        print(e)
        return
    
    # Initialize hand detector
    detector = HandDetector()
    
    print("Hand Detection Started!")
    print("- Open your hand: Door Opens (True)")
    print("- Close your hand: Door Closes (False)")
    print("- Press 'q' to quit")
    print("-" * 50)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand and get state
        frame, hand_state = detector.detect_hand(frame)
        
        # Print state to console
        if hand_state:
            print(f"Hand State: OPEN -> True", end='\r')
        else:
            print(f"Hand State: CLOSED/NOT DETECTED -> False", end='\r')
        
        # Display the frame
        cv2.imshow('Hand Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("\nHand Detection Stopped!")


if __name__ == "__main__":
    main()
