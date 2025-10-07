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
- `mediapipe` and `opencv-python` may be difficult to install on some Pi OS images. If you plan to run on a Raspberry Pi camera module, the script supports `libcamera` for better performance.

Using the Raspberry Pi camera (CSI)
----------------------------------

This repository supports the Raspberry Pi Camera Module (connected to the CSI pins) via `libcamera` Python bindings.

Basic steps (on the Raspberry Pi itself):

1. Update OS packages and install libcamera / python bindings (example for Raspberry Pi OS Bookworm/Bullseye):

```bash
sudo apt update
sudo apt full-upgrade -y
# Install libcap-dev (needed for python-prctl dependency if using pip)
sudo apt install -y libcap-dev
# Install libcamera, python bindings, and OpenCV
sudo apt install -y python3-libcamera python3-opencv libcamera-apps
```

2. Install Python packages in your virtualenv (or system Python):

```bash
# activate your venv, then:
pip install mediapipe opencv-python
# Note: on some Pi images mediapipe may need a prebuilt wheel or platform-specific install.
```

3. Run the demo using the Pi camera (script will use libcamera when `--use-picamera` is passed):

```bash
python face_mediapipe.py --use-picamera --model mesh
```

Notes and troubleshooting
- If `from libcamera import CameraManager` fails, install `python3-libcamera` as shown above.
- If you get "No module named 'libcamera'" error, install `python3-libcamera` as shown above.
- If you see color issues (frame looks blue/tinted), let me know — libcamera may return BGR instead of RGB and we can add a conversion.
- `mediapipe` installation on Pi can be the most painful part. If pip install fails, tell me the Pi model and OS (32-bit vs 64-bit) and I can provide precise instructions or a prebuilt wheel suggestion.
- If pip install of packages fails with "libcap development headers" error, install `libcap-dev` first as shown above, or use the OS packages instead of pip (recommended for Pi).
- **Recommended**: Use the OS packages for libcamera (`sudo apt install python3-libcamera`) instead of pip to avoid build issues.

If you'd like, I can expand this README with tested step-by-step Pi instructions for your specific Pi model.

.gitignore
- The repository includes a `.gitignore` which excludes `.venv/`, `__pycache__/`, snapshots and common Python/IDE artifacts.

If you want, I can also add a `requirements.txt` file or add picamera2 support.
