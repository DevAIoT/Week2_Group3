# face_mediapipe

Small demo that uses MediaPipe and OpenCV for face detection and face mesh.

Files
- `face_mediapipe.py` — main demo. Note: the script was renamed from `mediapipe.py` to avoid shadowing the installed `mediapipe` package.

Requirements
- Python 3.8+
- mediapipe
- opencv-python

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

Raspberry Pi notes
- `mediapipe` and `opencv-python` may be difficult to install on some Pi OS images. If you plan to run on a Raspberry Pi camera module, I can update the script to optionally use `picamera2`/libcamera for better performance — tell me your Pi model and OS if you'd like that.

.gitignore
- The repository includes a `.gitignore` which excludes `.venv/`, `__pycache__/`, snapshots and common Python/IDE artifacts.

If you want, I can also add a `requirements.txt` file or add picamera2 support.
