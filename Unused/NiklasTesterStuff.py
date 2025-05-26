from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("../YOLO12LFashionpedia100Epochs.pt")

img_rgb = cv2.imread("FolderTilTestData/person_0.jpg")
img_bgr = cv2.imread("FolderTilTestData/person_0.jpg")
img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Simulate BGR swap

print("RGB Detection:")
results_rgb = model.predict(img_rgb)
for item in results_rgb[0].boxes:
    print(model.names[int(item.cls.item())], item.conf.item())

print("\nBGR Detection:")
results_bgr = model.predict(img_bgr)
for item in results_bgr[0].boxes:
    print(model.names[int(item.cls.item())], item.conf.item())