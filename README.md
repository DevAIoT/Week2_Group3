# face_mediapipe

Small demo that uses MediaPipe and OpenCV for face detection and face mesh.

Files
- `face_mediapipe.py` — main demo. Note: the script was renamed from `mediapipe.py` to avoid shadowing the installed `mediapipe` package.
- `face_mesh_camera1.py` — standalone facial recognition script using camera 1 with face mesh features.

Requirements
- Python 3.8+
- mediapipe
- opencv-python
- numpy (for facial recognition features)

Install (recommended inside a virtual environment):

```powershell
# activate your venv first (Windows PowerShell)
# e.g. .\.venv\Scripts\Activate.ps1  (or use your preferred method)
pip install mediapipe opencv-python
```

Quick start

```powershell
# Run face detection (fast)
python face_mediapipe.py --model detection

# Run face mesh (landmarks)
python face_mediapipe.py --model mesh

# Use different camera (camera index 1) or a video/stream URL
python face_mediapipe.py --src 1 --model mesh
python face_mediapipe.py --src "rtsp://user:pass@host:554/stream" --model detection
```

Options
- `--src` : camera index (string) or path/URL (default: `0`)
- `--model` : `detection` or `mesh`
- `--width` / `--height` : requested capture resolution
- `--min-confidence` : minimum detection confidence (0.0–1.0)
- `--no-display` : run headless (no GUI)

Controls
- Press `q` to quit
- Press `s` to save a snapshot (files named `snapshot_<timestamp>.jpg`)

Facial Recognition
------------------

The `face_mesh_camera1.py` script includes facial recognition capabilities using MediaPipe face mesh landmarks.

### Features:
- Real-time face detection and landmark extraction
- Facial recognition using geometric features from face mesh
- Database of known faces stored in `known_faces.json`
- Visual feedback with colored bounding boxes (green for known, red for unknown)
- Confidence scores for recognition matches

### Usage:
```powershell
python face_mesh_camera1.py
```

### Controls:
- Press `q` to quit
- Press `s` to save a snapshot
- Press `a` to add a new face to the database (will prompt for name)
- Press `c` to clear the entire database

### Security Features:
- **Advanced Feature Extraction**: Uses 14+ facial landmarks for detailed feature vectors
- **Z-score Normalization**: Features are normalized for consistent comparison
- **85% Similarity Threshold**: Strict matching requirements (upgraded from 60%)
- **Multiple Facial Proportions**: Eye distance, mouth width, face shape ratios
- **Distance-based Verification**: Euclidean distance with conservative thresholds

### Database:
- Known faces are stored in `known_faces.json` in the script directory
- Each person can have multiple face samples for better recognition
- Database is automatically loaded on startup and saved when adding new faces

Raspberry Pi notes
- `mediapipe` and `opencv-python` may be difficult to install on some Pi OS images. If you plan to run on a Raspberry Pi camera module, the script supports `libcamera` for better performance.

Using the Raspberry Pi camera (CSI)
----------------------------------

This repository supports the Raspberry Pi Camera Module (connected to the CSI pins) via OpenCV with libcamera backend.

Basic steps (on the Raspberry Pi itself):

1. Update OS packages and install OpenCV with libcamera support (example for Raspberry Pi OS Bookworm/Bullseye):

```bash
sudo apt update
sudo apt full-upgrade -y
# Install OpenCV and libcamera
sudo apt install -y python3-opencv libcamera-apps
```

2. Install Python packages in your virtualenv (or system Python):

```bash
# activate your venv, then:
pip install mediapipe
# Note: on some Pi images mediapipe may need a prebuilt wheel. For Raspberry Pi 4/5, download from:
# https://github.com/nihui/mediapipe/releases (look for aarch64 wheel)
# Then pip install the .whl file
```

3. Run the demo using the Pi camera (script will try common CSI camera devices when `--use-picamera` is passed):

```bash
python face_mediapipe.py --use-picamera --model mesh
```

Notes and troubleshooting
- The script tries `/dev/video0`, `/dev/video10`, `/dev/video11`, `/dev/video12`, `/dev/video20` for the CSI camera.
- If the camera doesn't work, make sure libcamera is configured as the OpenCV backend (it should be by default on Bookworm).
- If you get protobuf errors with mediapipe, use a prebuilt wheel from the link above.
- If you see color issues (frame looks blue/tinted), let me know — we can add color conversion.
- `mediapipe` installation on Pi can be the most painful part. If pip install fails, use the prebuilt wheel for your Pi model.
- **Recommended**: Use the OS packages for OpenCV and libcamera to avoid build issues.

If you'd like, I can expand this README with tested step-by-step Pi instructions for your specific Pi model.

.gitignore
- The repository includes a `.gitignore` which excludes `.venv/`, `__pycache__/`, snapshots and common Python/IDE artifacts.

If you want, I can also add a `requirements.txt` file or add picamera2 support.
