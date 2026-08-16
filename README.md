# Hand Gesture Mouse Controller

A Python application that uses a webcam and MediaPipe hand tracking to control the mouse with hand gestures.

## Features

### Left hand

* Open palm — move the cursor
* Thumb + index finger pinch — hold the left mouse button
* Index finger + pinky finger gesture — scroll
* Fist — freeze the cursor

### Right hand

* Fist — right mouse click

## Requirements

* Python 3.10+
* Webcam
* Windows (PyAutoGUI is used for mouse control)

## Installation

```bash
git clone <your-repository-url>
cd hand-gesture-mouse

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Place `hand_landmarker.task` in the project folder next to `Code.py`.

## Run

```bash
python Code.py
```

Press `ESC` to exit the application.

## Dependencies

* OpenCV
* MediaPipe
* NumPy
* PyAutoGUI
