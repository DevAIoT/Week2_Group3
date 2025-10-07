#!/usr/bin/env python3
"""
simple_face_mesh.py

Simple standalone MediaPipe face mesh detection using camera source 1.
No command-line arguments - just runs with preset configuration.

Equivalent to: python face_mediapipe.py --src 1 --model mesh

Controls:
- Press 'q' to quit
- Press 's' to save a snapshot
"""
from __future__ import annotations

import time
import sys

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

    print(f"Opening camera {camera_index}...")
    
    try:
        cap = get_capture(camera_index, width, height)
        print(f"Camera {camera_index} opened successfully")
    except RuntimeError as e:
        print(f"Failed to open camera {camera_index}: {e}")
        print("Trying camera 0 as fallback...")
        try:
            camera_index = 0
            cap = get_capture(camera_index, width, height)
            print(f"Camera {camera_index} opened successfully")
        except RuntimeError as e2:
            print(f"Failed to open camera 0: {e2}")
            return 1

    # Initialize face mesh detector
    face_mesher = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.5
    )

    print("Face mesh detector initialized")
    print("Press 'q' to quit, 's' to save snapshot")

    prev_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame")
                break

            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process with face mesh
            results = face_mesher.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face mesh landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
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

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display
            cv2.imshow("Face Mesh - Camera 1", image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                print("Quit key pressed")
                break
            elif key == ord("s"):
                # Save snapshot
                ts = int(time.time())
                filename = f"snapshot_{ts}.jpg"
                cv2.imwrite(filename, image)
                print(f"Saved {filename}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if face_mesher is not None:
            face_mesher.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
