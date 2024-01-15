"""
Basic script that shows how to use the face detection model within OpenCV.
"""

import cv2
from cfg.detection import face_detection_cfg as cfg


def main():
    """
    Main function of the script.
    """
    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cfg.FACE_DETECTION_MODEL_PATH)

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

    # Display the output
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if cfg.SAVE:
        # Save the output
        cv2.imwrite(cfg.OUTPUT_PATH, img)


if __name__ == "__main__":
    main()