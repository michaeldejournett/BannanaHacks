# Banana_Cam — Run instructions

This folder contains a simple webcam-based demo (`BananaCam.py`) that captures frames and runs a placeholder AI detection function. `test_cam.py` is a minimal camera tester.

Quick steps (PowerShell on Windows):

1. Open PowerShell and change to this folder:

```powershell
cd C:\Users\rynow\Git\BannanaHacks\Banana_Cam
```

2. Create and activate a virtual environment, install dependencies, and run the camera app:

```powershell
# Create venv (only first time)
python -m venv .venv

# Activate the venv (PowerShell)
. .\.venv\Scripts\Activate.ps1

# Install required package
pip install --upgrade pip
pip install opencv-python

# Run the main app
python .\BananaCam.py
```

Alternatively you can use the helper script `run_bananacam.ps1` which does the above steps automatically (may require setting PowerShell execution policy or running with `-ExecutionPolicy Bypass`).

Troubleshooting
- If the camera fails to open, try a different device index (e.g. change `VideoCapture(0)` to `VideoCapture(1)`), or use the DirectShow backend on Windows:

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

- If you see an error importing `cv2`, ensure `opencv-python` was installed into the same Python used to run the script.
- If `run_bananacam.ps1` is blocked by policy, run it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bananacam.ps1
```

Notes
- `BananaCam.py` currently contains a placeholder `detect_banana()` that always returns False. Replace that with your model inference code when ready.
- `BananaCamYOLO.py` is empty in this copy — if you have a YOLO-based detector, put it here and call it from `process_frame()`.
