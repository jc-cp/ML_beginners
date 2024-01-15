"""
Basic script that shows how to use the face detection model within OpenCV.
"""
import sys
import os
import cv2

def manage_paths():
    """
    Add the 'cfg' directory to sys.path.
    """
    # Get the directory of the current script
    current_script_path = os.path.dirname(os.path.abspath(__file__))

    # Add the 'cfg' directory to sys.path
    cfg_path = os.path.join(current_script_path, '..', '..', 'cfg')
    absolute_cfg_path = os.path.abspath(cfg_path)
    sys.path.append(absolute_cfg_path)
    print("Path: ", absolute_cfg_path)

manage_paths()

import detection.face_detection_cfg as cfg


def main():
    """
    Main function of the script.
    """

    # Attempt to load the classifier
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    except Exception as error:
        print("Error loading classifier:", error)
        return

    # Load the image
    img = cv2.imread(cfg.IMAGE_PATH)

    if cfg.GRAY:
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    # Save the output
    cv2.imwrite(cfg.OUTPUT_PATH, img)


if __name__ == "__main__":
    main()