"""
face_mediapipe.py

Same MediaPipe demo as the original `mediapipe.py` but renamed so it does
not shadow the installed `mediapipe` package. Run this file instead:

  python face_mediapipe.py --model detection

"""
from __future__ import annotations

import argparse
import time
import sys
from typing import Optional

try:
    import cv2
except Exception:  # pragma: no cover - provides friendly message
    print("OpenCV (cv2) is required. Install with: pip install opencv-python")
    raise

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - provides friendly message
    print("mediapipe is required. Install with: pip install mediapipe")
    raise


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe face demo for Raspberry Pi / webcam")
    parser.add_argument("--src", type=str, default="1",
                        help="Video source. Camera index (0) or path to video file. Default=0")
    parser.add_argument("--model", choices=("detection", "mesh"), default="detection",
                        help="Which MediaPipe model to run: 'detection' (fast) or 'mesh' (landmarks)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum detection confidence (0-1)")
    parser.add_argument("--use-picamera", action="store_true",
                        help="Use the Raspberry Pi CSI camera (tries /dev/video0, /dev/video10, etc.)")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI display)")
    return parser.parse_args()


def get_capture(src: str, width: int, height: int, use_picamera: bool = False) -> cv2.VideoCapture:
    """Return an OpenCV VideoCapture for the given source.

    src may be a camera index (string containing digits) or a file path.
    If use_picamera is True, tries common CSI camera device paths.
    """
    if use_picamera:
        # Try common CSI camera device paths on Raspberry Pi
        camera_paths = ["/dev/video0", "/dev/video10", "/dev/video11", "/dev/video12", "/dev/video20"]
        for path in camera_paths:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                print(f"Opened CSI camera at {path}")
                break
        else:
            raise RuntimeError("Could not open any CSI camera device")
    elif src.isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(src)

    # Try to set resolution; drivers may ignore these calls
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")
    return cap


def main() -> int:
    args = parse_args()

    picam = None
    cap = None

    # Initialize capture: either Pi CSI camera (via OpenCV with libcamera backend) or USB webcam
    if args.use_picamera:
        cap = get_capture(args.src, args.width, args.height, use_picamera=True)
        picam = None
    else:
        cap = get_capture(args.src, args.width, args.height)
        picam = None

    # Choose model
    if args.model == "detection":
        face_detector = mp_face_detection.FaceDetection(min_detection_confidence=args.min_confidence,
                                                        model_selection=0)
        face_mesher = None
    else:
        face_detector = None
        face_mesher = mp_face_mesh.FaceMesh(static_image_mode=False,
                                            max_num_faces=2,
                                            refine_landmarks=True,
                                            min_detection_confidence=args.min_confidence,
                                            min_tracking_confidence=0.5)

    prev_time = time.time()
    try:
        while True:
            # Capture frame from the selected source
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame")
                break
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            if face_detector is not None:
                results = face_detector.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(image, detection)
            else:
                results = face_mesher.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show
            if not args.no_display:
                cv2.imshow("MediaPipe Face", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    # save snapshot
                    ts = int(time.time())
                    filename = f"snapshot_{ts}.jpg"
                    cv2.imwrite(filename, image)
                    print(f"Saved {filename}")
            else:
                # headless mode -- just loop; allow external interrupt
                pass

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if face_detector is not None:
            face_detector.close()
        if face_mesher is not None:
            face_mesher.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
