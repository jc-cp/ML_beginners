"""
Configuration file for face detection.
"""
from pathlib import Path

# Path to the face detection model

PROJECT_PATH = Path(__file__).resolve().parents[2]

IMAGE_PATH = str(PROJECT_PATH / "data/face.jpg")
GRAY = False
SAVE = False
OUTPUT_PATH = str(PROJECT_PATH / "output/face_detection.jpg")
