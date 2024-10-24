# Laser Pointer Detection Using OpenCV

This project implements a laser pointer detection system using OpenCV in Python. The system can detect a red laser pointer from a webcam feed and estimate the distance of the laser pointer from the camera. It is designed to detect laser pointers up to 2 meters away, with a tolerance of 10 cm.

## Features
- Detects red laser pointer spots in real-time using webcam video feed.
- Estimates the distance of the laser pointer from the camera.
- Capable of detecting laser spots up to a distance of 2 meters with a 10 cm tolerance.

## Requirements
To run this project, you will need the following:
- Python 3.x
- OpenCV (cv2 library)
- NumPy

You can install the dependencies using pip:
```bash
pip install opencv-python numpy
```
Run the script:
```bash
python laser_pointer_detection.py
