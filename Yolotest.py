from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/last.pt")  # path to your model weights

# Predict on a folder of images or a single image
results = model.predict(source="testdata",  # can be a folder or single file
                        conf=0.10,  # confidence threshold, adjust as needed
                        save=True)  # save visualized predictions (bounding boxes) to runs/predict
