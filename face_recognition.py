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
    """Extract rotation-invariant features from face mesh landmarks for recognition.
    Uses 42 comprehensive landmarks covering all major facial features.
    Rotation-invariant by using distances and ratios instead of absolute positions."""
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

    # Define reference points for face coordinate system
    nose_tip = points[1]
    left_eye = points[33]
    right_eye = points[263]
    
    # Create rotation-invariant coordinate system
    # Use eye-to-eye vector as X-axis reference
    eye_vector = right_eye - left_eye
    eye_distance = np.linalg.norm(eye_vector)
    
    # Normalize by eye distance to make scale-invariant
    if eye_distance == 0:
        eye_distance = 1.0  # Avoid division by zero
    
    features = []
    
    # ROTATION-INVARIANT APPROACH: Use distances between key points
    # Instead of relative positions, use pairwise distances normalized by eye distance
    
    # Calculate distances from nose tip to all other key points (normalized)
    for idx in key_points:
        if idx != 1:  # Skip nose tip itself
            point = points[idx]
            distance = np.linalg.norm(point - nose_tip) / eye_distance
            features.append(distance)
    
    # Add pairwise distances between important facial features (normalized)
    # These are rotation-invariant geometric measurements
    
    # Eye measurements (normalized by eye distance)
    left_eye_center = points[145]
    right_eye_center = points[374]
    eyes_to_nose = np.linalg.norm(left_eye_center - nose_tip) / eye_distance
    features.append(eyes_to_nose)
    
    # Mouth measurements
    mouth_left = points[61]
    mouth_right = points[291]
    mouth_center = (mouth_left + mouth_right) / 2
    mouth_to_nose = np.linalg.norm(mouth_center - nose_tip) / eye_distance
    features.append(mouth_to_nose)

    # Add comprehensive RATIOS and normalized distances (rotation-invariant)
    # All measurements normalized by eye distance for scale invariance
    
    # 1. Mouth width ratio (normalized by eye distance)
    mouth_width = np.linalg.norm(mouth_right - mouth_left) / eye_distance
    features.append(mouth_width)
    
    # 2. Eye-to-mouth vertical distance ratio
    eye_center = (left_eye + right_eye) / 2
    eye_mouth_distance = np.linalg.norm(eye_center - mouth_center) / eye_distance
    features.append(eye_mouth_distance)
    
    # 3. Face width ratio (cheek to cheek / eye distance)
    left_cheek = points[234]
    right_cheek = points[454]
    face_width_ratio = np.linalg.norm(left_cheek - right_cheek) / eye_distance
    features.append(face_width_ratio)
    
    # 4. Face height ratio (forehead to chin / eye distance)
    forehead = points[10]
    chin = points[152]
    face_height_ratio = np.linalg.norm(forehead - chin) / eye_distance
    features.append(face_height_ratio)
    
    # 5. Nose length ratio (bridge to tip / eye distance)
    nose_bridge = points[0]
    nose_length_ratio = np.linalg.norm(nose_bridge - nose_tip) / eye_distance
    features.append(nose_length_ratio)
    
    # 6. Left eye height ratio
    left_eye_height_ratio = np.linalg.norm(points[160] - points[144]) / eye_distance
    features.append(left_eye_height_ratio)
    
    # 7. Right eye height ratio
    right_eye_height_ratio = np.linalg.norm(points[387] - points[373]) / eye_distance
    features.append(right_eye_height_ratio)
    
    # 8. Eyebrow span ratio
    left_eyebrow = points[70]
    right_eyebrow = points[300]
    eyebrow_span_ratio = np.linalg.norm(left_eyebrow - right_eyebrow) / eye_distance
    features.append(eyebrow_span_ratio)
    
    # 9. Mouth height ratio (upper to lower lip / eye distance)
    upper_lip = points[13]
    lower_lip = points[14]
    mouth_height_ratio = np.linalg.norm(upper_lip - lower_lip) / eye_distance
    features.append(mouth_height_ratio)
    
    # 10. Nose-to-chin ratio
    nose_chin_ratio = np.linalg.norm(nose_tip - chin) / eye_distance
    features.append(nose_chin_ratio)
    
    # 11. Face aspect ratio (height / width)
    face_width = np.linalg.norm(left_cheek - right_cheek)
    face_height = np.linalg.norm(forehead - chin)
    if face_width > 0:
        aspect_ratio = face_height / face_width
    else:
        aspect_ratio = 1.0
    features.append(aspect_ratio)
    
    # 12. Eye separation ratio (inner to outer distance)
    left_eye_inner = points[133]
    left_eye_outer = points[33]
    right_eye_inner = points[362]
    right_eye_outer = points[263]
    left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner) / eye_distance
    right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner) / eye_distance
    features.append(left_eye_width)
    features.append(right_eye_width)

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


def find_best_match(features: np.ndarray, known_faces: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float]:
    """Find the best matching face using adaptive thresholds and statistical analysis.
    
    Args:
        features: Feature vector of the face to match
        known_faces: Database of known faces
    
    Returns:
        Tuple of (name, confidence) or (None, 0.0) if no match
    """
    if not known_faces:
        return None, 0.0
    
    # Calculate distances to ALL samples of ALL people
    person_matches = {}
    all_distances = []  # Track all distances for adaptive thresholding
    
    for name, encodings in known_faces.items():
        distances = []
        for known_features in encodings:
            distance = np.linalg.norm(features - known_features)
            distances.append(distance)
            all_distances.append(distance)
        
        # Use BEST 3 matches (or all if less than 3) - most discriminative
        best_distances = sorted(distances)[:min(3, len(distances))]
        avg_best_distance = np.mean(best_distances)
        median_distance = np.median(distances)
        min_distance = min(distances)
        
        person_matches[name] = {
            'min_distance': min_distance,
            'avg_best_distance': avg_best_distance,
            'median_distance': median_distance,
            'all_distances': distances,
            'num_samples': len(distances)
        }
    
    # Find person with minimum distance
    best_match = None
    best_min_distance = float('inf')
    
    for name, stats in person_matches.items():
        if stats['min_distance'] < best_min_distance:
            best_min_distance = stats['min_distance']
            best_match = name
    
    if not best_match:
        return None, 0.0
    
    # ADAPTIVE THRESHOLDING based on dataset statistics
    best_stats = person_matches[best_match]
    
    # Calculate statistics for dataset quality
    num_people = len(known_faces)
    total_samples = sum(stats['num_samples'] for stats in person_matches.values())
    avg_samples = total_samples / num_people
    
    # Base thresholds (distance-based)
    if avg_samples >= 20:
        # Good dataset - strict threshold
        distance_threshold = 0.35
        min_votes = 0.6  # 60% of top-3 must match
    elif avg_samples >= 10:
        # Decent dataset - medium threshold
        distance_threshold = 0.45
        min_votes = 0.5
    else:
        # Poor dataset - lenient threshold
        distance_threshold = 0.55
        min_votes = 0.4
    
    # Check 1: Minimum distance must be below threshold
    if best_min_distance > distance_threshold:
        return None, 0.0
    
    # Check 2: Average of best 3 matches must be good
    if best_stats['avg_best_distance'] > distance_threshold * 1.3:
        return None, 0.0
    
    # Check 3: Voting - how many samples agree?
    good_matches = sum(1 for d in best_stats['all_distances'] if d < distance_threshold * 1.5)
    vote_ratio = good_matches / best_stats['num_samples']
    
    if vote_ratio < min_votes:
        return None, 0.0
    
    # Check 4: Distinctiveness - must be clearly different from others
    if num_people > 1:
        # Get second-best match
        second_best_distance = float('inf')
        for name, stats in person_matches.items():
            if name != best_match:
                if stats['min_distance'] < second_best_distance:
                    second_best_distance = stats['min_distance']
        
        # Require significant gap (at least 25% better than second place)
        margin = (second_best_distance - best_min_distance) / (best_min_distance + 0.01)
        
        if margin < 0.25:  # Not distinctive enough
            return None, 0.0
    else:
        # Single person in database - be EXTRA careful
        # Require very close match since there's no comparison
        if best_min_distance > distance_threshold * 0.7:
            return None, 0.0
        
        if best_stats['avg_best_distance'] > distance_threshold * 0.9:
            return None, 0.0
    
    # Calculate confidence score (0-1)
    # Based on: distance quality + voting confidence + distinctiveness
    distance_score = max(0, 1 - (best_min_distance / distance_threshold))
    vote_score = vote_ratio
    
    if num_people > 1:
        # Include distinctiveness in confidence
        margin = (second_best_distance - best_min_distance) / (best_min_distance + 0.01)
        distinct_score = min(1.0, margin / 0.5)  # Max out at 50% margin
        confidence = (distance_score * 0.5 + vote_score * 0.3 + distinct_score * 0.2)
    else:
        # Single person - heavily weight distance quality
        confidence = (distance_score * 0.7 + vote_score * 0.3)
    
    return best_match, confidence


def add_face_to_database(name: str, features: np.ndarray, known_faces: Dict[str, List[np.ndarray]]):
    """Add a face to the known faces database."""
    if name not in known_faces:
        known_faces[name] = []
    known_faces[name].append(features)
    print(f"Added face for {name}. Total faces for {name}: {len(known_faces[name])}")


def auto_dataset_collector(cap, face_mesher, known_faces: Dict[str, List[np.ndarray]], 
                           name: str, duration: int = 15) -> bool:
    """
    Automatically collect face samples while user moves their head naturally.
    
    Args:
        cap: Camera capture object
        face_mesher: MediaPipe face mesh detector
        known_faces: Dictionary to store collected samples
        name: Person's name for the dataset
        duration: Duration in seconds to collect samples (default 15)
    
    Returns:
        bool: True if collection was successful, False if cancelled
    """
    collected_samples = []
    frame_interval = 10  # Capture every 10 frames (~0.3 seconds at 30fps)
    frame_count = 0
    
    start_time = time.time()
    last_capture_time = 0
    
    print("\n" + "="*70)
    print("🤖 AUTOMATIC DATASET COLLECTOR")
    print("="*70)
    print(f"   Collecting samples for: {name}")
    print(f"   Duration: {duration} seconds")
    print(f"   Move your head around naturally!")
    print(f"   Press 'ESC' to cancel at any time")
    print("="*70 + "\n")
    
    window_name = "Auto Dataset Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    while True:
        # Check if time is up
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            break
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Camera error")
            return False
        
        frame_count += 1
        
        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesher.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        height, width = image.shape[:2]
        
        # Draw face mesh if detected
        face_detected = False
        current_features = None
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Extract features and capture every N frames
            if frame_count % frame_interval == 0:
                current_features = extract_face_features(face_landmarks)
                collected_samples.append(current_features)
                print(f"  ✓ Captured sample {len(collected_samples)} at {elapsed_time:.1f}s")
        
        # Calculate time remaining
        time_remaining = int(duration - elapsed_time)
        
        # Overlay instructions
        overlay = image.copy()
        
        # Top banner - dark background
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Time progress bar
        progress = elapsed_time / duration
        bar_width = width - 40
        bar_x = 20
        bar_y = 20
        bar_height = 30
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Time text
        time_text = f"Time: {int(elapsed_time)}s / {duration}s  |  Samples: {len(collected_samples)}"
        cv2.putText(image, time_text, (bar_x + 5, bar_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Main instruction - large text
        main_instruction = "Move your head around naturally!"
        cv2.putText(image, main_instruction, (20, 75), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2)
        
        # Sub instructions
        sub_instructions = [
            "• Turn left and right",
            "• Look up and down", 
            "• Try different expressions"
        ]
        y_pos = 110
        for instruction in sub_instructions:
            cv2.putText(image, instruction, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
        
        # Status message
        if not face_detected:
            status_color = (0, 0, 255)  # Red
            status_text = "⚠ NO FACE DETECTED"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = f"✓ Face detected - Capturing every {frame_interval} frames"
        
        cv2.putText(image, status_text, (20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Bottom instruction
        cv2.rectangle(image, (0, height - 40), (width, height), (0, 0, 0), -1)
        cv2.putText(image, f"Time remaining: {time_remaining}s  |  Press ESC to cancel", (20, height - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n✗ Collection cancelled by user")
            cv2.destroyWindow(window_name)
            return False
    
    # Collection complete
    print("\n" + "="*70)
    print("✓ COLLECTION COMPLETE!")
    print("="*70)
    
    # Add all samples to database
    if name not in known_faces:
        known_faces[name] = []
    known_faces[name].extend(collected_samples)
    
    print(f"✓ Added {len(collected_samples)} samples for {name}")
    print(f"  Total samples for {name}: {len(known_faces[name])}")
    print("="*70 + "\n")
    
    # Show success screen for 2 seconds
    success_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(success_frame, "DATASET COMPLETE!", (150, 250), 
               cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    cv2.putText(success_frame, f"{len(collected_samples)} samples collected for {name}", (180, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(success_frame, "Returning to recognition mode...", (200, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    cv2.imshow(window_name, success_frame)
    cv2.waitKey(2000)
    
    cv2.destroyWindow(window_name)
    return True


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
    
    # Analyze model quality
    model_quality = "EXCELLENT"
    min_samples = 999
    needs_improvement = []
    
    if known_faces:
        for name, samples in known_faces.items():
            num_samples = len(samples)
            print(f"  - {name}: {num_samples} sample(s)", end="")
            
            if num_samples >= 30:
                print(" ✓ Excellent")
            elif num_samples >= 20:
                print(" ✓ Very Good")
            elif num_samples >= 15:
                print(" ✓ Good")
            elif num_samples >= 10:
                print(" ⚠ Acceptable")
                if model_quality == "EXCELLENT":
                    model_quality = "GOOD"
                needs_improvement.append(name)
            else:
                print(f" ❌ Poor - Add {15 - num_samples} more!")
                model_quality = "POOR"
                needs_improvement.append(name)
            
            min_samples = min(min_samples, num_samples)
    
    # Display model quality assessment
    print("\n" + "="*70)
    print("📊 MODEL QUALITY ASSESSMENT")
    print("="*70)
    
    if model_quality == "EXCELLENT":
        print("✓ Status: EXCELLENT - High accuracy expected")
        print("  Recognition threshold: 82% (strict)")
    elif model_quality == "GOOD":
        print("⚠ Status: GOOD - Decent accuracy, but can improve")
        print("  Recognition threshold: 82% (strict)")
        print(f"  Recommendation: Add more samples for: {', '.join(needs_improvement)}")
    else:
        print("❌ Status: POOR - Low accuracy, false positives likely")
        print("  Recognition threshold: 82% (strict, may still have errors)")
        print(f"  ⚠️  URGENT: Add 15+ samples per person for reliable recognition")
        if needs_improvement:
            print(f"  People needing more samples: {', '.join(needs_improvement)}")
    
    print("\n💡 ACCURACY IMPROVEMENT TIPS:")
    print("  1. Use 'a' for auto-collection (15-30 seconds recommended)")
    print("  2. Move head naturally during collection (all angles)")
    print("  3. Collect 20-30 samples per person for best results")
    print("  4. Add multiple people to test distinctiveness")
    print("  5. Watch confidence scores - HIGH (>90%) = very confident")
    print("="*70 + "\n")

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
    print("="*70)
    print("📋 KEYBOARD CONTROLS:")
    print("  'q' = Quit program")
    print("  's' = Save snapshot")
    print("  'a' = 🤖 AUTO DATASET COLLECTOR (guided multi-pose capture)")
    print("  'c' = Clear entire database")
    print("="*70)
    print("👀 IMPORTANT: Window will appear as 'Face Recognition - Camera 1'")
    print("   If window doesn't appear, check your taskbar or try Alt+Tab")
    print("   The script will run continuously until you press 'q' to quit")
    print("\n💡 TIP: Press 'a' to automatically collect 15+ samples with guided poses!\n")

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

                    # Draw recognition result with confidence-based coloring
                    if name:
                        # Known face - color based on confidence
                        if confidence >= 0.75:
                            color = (0, 255, 0)  # Bright green - very confident
                            conf_label = "HIGH"
                        elif confidence >= 0.60:
                            color = (0, 200, 100)  # Medium green - confident
                            conf_label = "GOOD"
                        elif confidence >= 0.45:
                            color = (0, 150, 150)  # Yellow-green - medium confidence
                            conf_label = "MEDIUM"
                        else:
                            color = (0, 100, 200)  # Blue-green - low confidence
                            conf_label = "LOW"
                        
                        text = f"{name}"
                        conf_text = f"{confidence*100:.0f}% ({conf_label})"
                    else:
                        # Unknown face - red
                        color = (0, 0, 255)
                        text = f"Unknown"
                        conf_text = f"No match"

                    # Draw bounding box (thicker for high confidence)
                    box_thickness = 3 if (name and confidence >= 0.90) else 2
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, box_thickness)

                    # Draw name label with background
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(image, (x_min, y_min - 35), (x_min + text_size[0] + 10, y_min), color, -1)
                    cv2.putText(image, text, (x_min + 5, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show detailed confidence info
                    if name:
                        num_samples = len(known_faces.get(name, []))
                        detail_text = f"{conf_text} | Samples: {num_samples}"
                    else:
                        detail_text = conf_text
                    
                    cv2.putText(image, detail_text, (x_min, y_max + 20),
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
                # Auto dataset collector mode
                print("\n" + "="*70)
                print("🤖 AUTOMATIC DATASET COLLECTOR")
                print("="*70)
                current_name = input("Enter name for dataset collection (or press Enter to cancel): ").strip()
                if current_name:
                    duration_input = input(f"How many seconds to record? (default 15, recommended 10-20): ").strip()
                    try:
                        duration = int(duration_input) if duration_input else 15
                        duration = max(5, min(60, duration))  # Clamp between 5 and 60 seconds
                    except ValueError:
                        duration = 15
                    
                    print(f"\n🎯 Recording for {duration} seconds for: {current_name}")
                    print("   Just move your head around naturally!")
                    print("   The system will capture frames automatically\n")
                    
                    # Run auto collector
                    success = auto_dataset_collector(cap, face_mesher, known_faces, current_name, duration)
                    
                    if success:
                        save_known_faces(known_faces)
                        print("💾 Dataset saved successfully!")
                    else:
                        print("⚠️  Dataset collection was cancelled or incomplete")
                    
                    print("\n🎥 Resuming video stream...")
                    print("="*70 + "\n")
                else:
                    print("✗ Cancelled - no name entered\n")
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
