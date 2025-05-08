import os
import cv2
from ultralytics import YOLO

# Initialize YOLO model once (so you don't repeatedly load weights)
model = YOLO("yolov8n.pt")

def find_people_in_image(opencv_image, confidence_threshold=0.5):
    """
    Returns: (person_count, list_of_cropped_images)
    """
    model = YOLO("yolov8n.pt")
    results = model.predict(opencv_image)
    cropped_images = []
    person_count = 0

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_images.append(opencv_image[y1:y2, x1:x2])
                person_count += 1

    return person_count, cropped_images
