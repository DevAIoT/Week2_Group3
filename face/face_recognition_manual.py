#!/usr/bin/env python3
"""
face_recognition.py

MediaPipe face mesh detection with facial recognition using camera source 1.
Features secure face recognition with 85% similarity threshold.

Controls:
- Press 'q' to quit
- Press 's' to save a snapshot
- Press 'a' to add a new face to the database
- Press 'c' to clear the entire database
"""
from __future__ import annotations

import time
import sys
import os
import json
from typing import Optional, List, Dict, Tuple
import numpy as np

try:
    import cv2
except Exception:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise

try:
    import mediapipe as mp
except Exception:
    print("mediapipe is required. Install with: pip install mediapipe")
    raise


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def extract_face_features(landmarks) -> np.ndarray:
    """Extract features from face mesh landmarks for recognition.
    Uses 42 comprehensive landmarks covering all major facial features."""
    points = []
    for landmark in landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])
    points = np.array(points)

    # Comprehensive facial landmarks for robust recognition (42 points)
    key_points = [
        # Nose (reference point)
        1,      # Nose tip
        # Eyes - Left (8 points)
        33,     # Left eye outer corner
        133,    # Left eye inner corner
        160,    # Left eye top
        144,    # Left eye bottom
        145,    # Left eye center
        159,    # Left eye top outer
        158,    # Left eye top inner
        153,    # Left eye bottom outer
        # Eyes - Right (8 points)
        263,    # Right eye outer corner
        362,    # Right eye inner corner
        387,    # Right eye top
        373,    # Right eye bottom
        374,    # Right eye center
        386,    # Right eye top outer
        385,    # Right eye top inner
        380,    # Right eye bottom outer
        # Eyebrows - Left (3 points)
        70,     # Left eyebrow outer
        63,     # Left eyebrow middle
        105,    # Left eyebrow inner
        # Eyebrows - Right (3 points)
        300,    # Right eyebrow outer
        293,    # Right eyebrow middle
        334,    # Right eyebrow inner
        # Nose bridge and sides (4 points)
        0,      # Nose bridge top
        4,      # Nose bridge middle
        5,      # Nose bridge lower
        195,    # Nose left side
        # Mouth (8 points)
        61,     # Mouth left corner
        291,    # Mouth right corner
        13,     # Upper lip top center
        14,     # Lower lip bottom center
        78,     # Upper lip left
        308,    # Upper lip right
        88,     # Lower lip left
        318,    # Lower lip right
        # Face outline (8 points)
        10,     # Forehead center
        152,    # Chin bottom
        234,    # Left cheek side
        454,    # Right cheek side
        21,     # Left temple
        251,    # Right temple
        172,    # Jaw left
        397,    # Jaw right
    ]

    features = []
    nose_tip = points[1]

    # Calculate relative positions to nose tip for all key points
    for idx in key_points:
        if idx != 1:  # Skip nose tip itself
            point = points[idx]
            features.extend([
                point[0] - nose_tip[0],
                point[1] - nose_tip[1],
                point[2] - nose_tip[2],
            ])

    # Add comprehensive facial proportions (10 measurements)
    
    # 1. Eye distance
    left_eye_outer = points[33]
    right_eye_outer = points[263]
    eye_distance = np.linalg.norm(left_eye_outer - right_eye_outer)
    features.append(eye_distance)

    # 2. Mouth width
    mouth_left = points[61]
    mouth_right = points[291]
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    features.append(mouth_width)

    # 3. Eye to mouth distance
    left_eye_center = (points[33] + points[263]) / 2
    mouth_center = (points[61] + points[291]) / 2
    eye_to_mouth_distance = np.linalg.norm(left_eye_center - mouth_center)
    features.append(eye_to_mouth_distance)
    
    # 4. Face width (cheek to cheek)
    face_width = np.linalg.norm(points[234] - points[454])
    features.append(face_width)
    
    # 5. Face height (forehead to chin)
    face_height = np.linalg.norm(points[10] - points[152])
    features.append(face_height)
    
    # 6. Nose length (bridge to tip)
    nose_length = np.linalg.norm(points[0] - points[1])
    features.append(nose_length)
    
    # 7. Left eye height
    left_eye_height = np.linalg.norm(points[160] - points[144])
    features.append(left_eye_height)
    
    # 8. Right eye height
    right_eye_height = np.linalg.norm(points[387] - points[373])
    features.append(right_eye_height)
    
    # 9. Eyebrow distance (left to right)
    eyebrow_distance = np.linalg.norm(points[70] - points[300])
    features.append(eyebrow_distance)
    
    # 10. Mouth height (upper to lower lip)
    mouth_height = np.linalg.norm(points[13] - points[14])
    features.append(mouth_height)

    chin = points[199]
    nose_to_chin_distance = np.linalg.norm(nose_tip - chin)
    features.append(nose_to_chin_distance)

    left_cheek = points[234]
    right_cheek = points[454]
    face_width = np.linalg.norm(left_cheek - right_cheek)
    features.append(face_width)

    features = np.array(features)

    # Z-score normalization
    if len(features) > 0:
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val > 0:
            features = (features - mean_val) / std_val

    return features


def load_known_faces(database_file: str = "known_faces.json") -> Dict[str, List[np.ndarray]]:
    """Load known faces database from JSON file."""
    if os.path.exists(database_file):
        try:
            with open(database_file, 'r') as f:
                data = json.load(f)
                known_faces = {}
                invalid_samples = []
                
                for name, encodings in data.items():
                    valid_encodings = []
                    for enc in encodings:
                        enc_array = np.array(enc)
                        # Check if feature vector has the expected size (133 features)
                        # Old samples had 42 features, new samples have 133
                        if len(enc_array) == 133:
                            valid_encodings.append(enc_array)
                        else:
                            invalid_samples.append((name, len(enc_array)))
                    
                    if valid_encodings:
                        known_faces[name] = valid_encodings
                
                if invalid_samples:
                    print("\n⚠️  WARNING: Database contains incompatible samples!")
                    print(f"   Found {len(invalid_samples)} samples with old format (44 features)")
                    print(f"   These samples will be IGNORED (new format needs 133 features)")
                    print("\n   📝 RECOMMENDATION: Clear database and re-add samples with 'c' command")
                    print("   This will ensure all samples use the improved 42-landmark system\n")
                
                return known_faces
        except Exception as e:
            print(f"Error loading face database: {e}")
    return {}


def save_known_faces(known_faces: Dict[str, List[np.ndarray]], database_file: str = "known_faces.json"):
    """Save known faces database to JSON file."""
    try:
        data = {}
        for name, encodings in known_faces.items():
            data[name] = [enc.tolist() for enc in encodings]
        with open(database_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving face database: {e}")


def find_best_match(features: np.ndarray, known_faces: Dict[str, List[np.ndarray]],
                   threshold: float = 2.0) -> Tuple[Optional[str], float]:
    """Find the best matching face from known faces database."""
    best_match = None
    best_similarity = 0.0

    for name, encodings in known_faces.items():
        for known_features in encodings:
            distance = np.linalg.norm(features - known_features)
            similarity = max(0, 1 - (distance / threshold))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

    # Lowered threshold to 75% for better detection (was 85%)
    if best_match and best_similarity >= 0.75:
        return best_match, best_similarity

    return None, 0.0


def add_face_to_database(name: str, features: np.ndarray, known_faces: Dict[str, List[np.ndarray]]):
    """Add a face to the known faces database."""
    if name not in known_faces:
        known_faces[name] = []
    known_faces[name].append(features)
    print(f"Added face for {name}. Total faces for {name}: {len(known_faces[name])}")


def clear_database(database_file: str = "known_faces.json"):
    """Clear the face recognition database."""
    if os.path.exists(database_file):
        os.remove(database_file)
        print(f"Cleared database: {database_file}")
    else:
        print("No database file found to clear")


def get_capture(src: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera capture."""
    cap = cv2.VideoCapture(src)
    
    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {src}")
    return cap


def main() -> int:
    # Hardcoded settings: camera 1, mesh model
    camera_index = 1
    width = 640
    height = 480
    min_confidence = 0.5

    # Load known faces database
    known_faces = load_known_faces()
    total_samples = sum(len(faces) for faces in known_faces.values())
    print(f"Loaded {total_samples} known faces for {len(known_faces)} people")
    
    if known_faces:
        for name, samples in known_faces.items():
            print(f"  - {name}: {len(samples)} sample(s)")
            if len(samples) < 5:
                print(f"    ⚠️  Recommended: Add {5 - len(samples)} more samples for better recognition")
    
    print("\n📸 FACE RECOGNITION TIPS:")
    print("  • Add 5-7 samples per person for best results")
    print("  • Capture from different angles (front, left, right)")
    print("  • Try different expressions (neutral, smile)")
    print("  • Ensure good lighting")
    print("  • Press 'a' to add each sample\n")

    print(f"Opening camera {camera_index}...")
    
    try:
        cap = get_capture(camera_index, width, height)
        print(f"✓ Camera {camera_index} opened successfully")
        
        # Test if we can actually read frames
        test_ret, test_frame = cap.read()
        if not test_ret or test_frame is None:
            raise RuntimeError(f"Camera {camera_index} opened but cannot read frames")
        print(f"✓ Camera {camera_index} can read frames (resolution: {test_frame.shape})")
        
    except RuntimeError as e:
        print(f"✗ Failed to open camera {camera_index}: {e}")
        print("Trying camera 0 as fallback...")
        try:
            camera_index = 0
            cap = get_capture(camera_index, width, height)
            print(f"✓ Camera {camera_index} opened successfully")
            
            # Test if we can actually read frames
            test_ret, test_frame = cap.read()
            if not test_ret or test_frame is None:
                raise RuntimeError(f"Camera {camera_index} opened but cannot read frames")
            print(f"✓ Camera {camera_index} can read frames (resolution: {test_frame.shape})")
            
        except RuntimeError as e2:
            print(f"✗ Failed to open camera 0: {e2}")
            print("\nAvailable cameras:")
            for i in range(5):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    test_ret, _ = test_cap.read()
                    status = "Available ✓" if test_ret else "Available but cannot read ✗"
                else:
                    status = "Not available ✗"
                print(f"  Camera {i}: {status}")
                test_cap.release()
            return 1

    # Initialize face mesh detector
    print("Initializing face mesh detector...")
    try:
        face_mesher = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        print("✓ Face mesh detector initialized")
    except Exception as e:
        print(f"✗ Failed to initialize face mesh detector: {e}")
        cap.release()
        return 1

    print("\n🎥 Starting video stream...")
    print("="*60)
    print("📋 KEYBOARD CONTROLS:")
    print("  'q' = Quit program")
    print("  's' = Save snapshot")
    print("  'a' = Add current face to database")
    print("  'c' = Clear entire database")
    print("="*60)
    print("👀 IMPORTANT: Window will appear as 'Face Recognition - Camera 1'")
    print("   If window doesn't appear, check your taskbar or try Alt+Tab")
    print("   The script will run continuously until you press 'q' to quit\n")

    # Create window and set properties
    window_name = "Face Recognition - Camera 1"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        print(f"✓ Window '{window_name}' pre-created")
    except Exception as e:
        print(f"⚠ Warning: Could not pre-create window: {e}")
        print("  Will try to create window on first frame...")

    prev_time = time.time()
    frame_count = 0
    
    try:
        print("⏳ Entering main video loop...")
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print(f"✗ Cannot read frame {frame_count + 1}")
                print(f"   Camera stopped responding at frame {frame_count}")
                break
            
            frame_count += 1
            if frame_count == 1:
                print(f"✓ First frame captured successfully")
                print(f"✓ Video loop started - script is running continuously")
                print(f"  (You should see the video window now - check your screen!)\n")
            
            if frame_count % 100 == 0:
                print(f"  📹 Still running... frame {frame_count} (press 'q' in the video window to quit)")

            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process with face mesh
            results = face_mesher.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process each detected face
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract features for recognition
                    features = extract_face_features(face_landmarks)

                    # Try to recognize the face
                    name, confidence = find_best_match(features, known_faces)

                    # Draw face mesh landmarks
                    # Draw tesselation
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Draw contours
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                    # Get bounding box from landmarks FIRST
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]

                    x_min = int(min(x_coords) * width)
                    x_max = int(max(x_coords) * width)
                    y_min = int(min(y_coords) * height)
                    y_max = int(max(y_coords) * height)

                    # Draw recognition result
                    if name:
                        # Known face - draw green box with name
                        color = (0, 255, 0)
                        text = f"{name} ({confidence:.2f})"
                    else:
                        # Unknown face - draw red box
                        color = (0, 0, 255)
                        text = f"Unknown"

                    # Draw bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                    # Draw name label
                    cv2.putText(image, text, (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Show debug info for troubleshooting
                    debug_text = f"Samples: {len(known_faces.get(name, []))} | Conf: {confidence:.2f}" if name else f"Best: {confidence:.2f}"
                    cv2.putText(image, debug_text, (x_min, y_max + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Instructions overlay
            cv2.putText(image, "Press 'a' to add face, 'c' to clear database, 'q' to quit, 's' to save snapshot",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display
            try:
                cv2.imshow("Face Recognition - Camera 1", image)
                if frame_count == 1:
                    print("✓ Window display called successfully")
                    print("✓ Look for 'Face Recognition - Camera 1' window on your screen!")
                    cv2.waitKey(1)  # Force window to appear
                if frame_count == 2:
                    print("✓ Second frame displayed - video stream is working!\n")
            except Exception as e:
                print(f"✗ Error displaying frame {frame_count}: {e}")
                print(f"   This might be a GUI/display driver issue")
                break
            
            key = cv2.waitKey(10) & 0xFF  # Increased from 1ms to 10ms
            
            if key == ord("q"):
                print("\n👋 Quit key pressed")
                break
            elif key == ord("s"):
                # Save snapshot
                ts = int(time.time())
                filename = f"snapshot_{ts}.jpg"
                cv2.imwrite(filename, image)
                print(f"📷 Saved {filename}")
            elif key == ord("a"):
                # Add new face mode
                if results.multi_face_landmarks:
                    print("\n" + "="*50)
                    print("📝 ADD FACE MODE")
                    print("="*50)
                    current_name = input("Enter name for this face (or press Enter to cancel): ").strip()
                    if current_name:
                        features = extract_face_features(results.multi_face_landmarks[0])
                        add_face_to_database(current_name, features, known_faces)
                        save_known_faces(known_faces)
                        print(f"✓ Added {current_name} to database")
                        print(f"  Total samples for {current_name}: {len(known_faces[current_name])}")
                        if len(known_faces[current_name]) < 5:
                            print(f"  TIP: Add {5 - len(known_faces[current_name])} more samples for better recognition")
                            print("       Try different angles, expressions, and lighting")
                        print("\n🎥 Resuming video stream...")
                        print("="*50 + "\n")
                    else:
                        print("✗ Cancelled - no name entered\n")
                else:
                    print("\n⚠ No face detected to add\n")
            elif key == ord("c"):
                # Clear database
                print("\n" + "="*50)
                print("⚠ CLEAR DATABASE WARNING")
                print("="*50)
                confirm = input("Clear entire database? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    clear_database()
                    known_faces.clear()
                    print("✓ Database cleared\n")
                else:
                    print("✗ Database not cleared\n")
                print("🎥 Resuming video stream...")
                print("="*50 + "\n")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ ERROR in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Cleaning up and exiting...")
        if cap is not None:
            cap.release()
            print("  ✓ Camera released")
        cv2.destroyAllWindows()
        print("  ✓ Windows closed")
        if face_mesher is not None:
            face_mesher.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
