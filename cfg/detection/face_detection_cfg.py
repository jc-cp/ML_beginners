"""
Configuration file for face detection.
"""
from pathlib import Path

# Path to the face detection model
FACE_DETECTION_MODEL_PATH = Path("models/face_detection.xml")
IMAGE_PATH = Path("data/face.jpg")
GRAY = False
SAVE = False
OUTPUT_PATH = Path("output/face_detection.jpg")
