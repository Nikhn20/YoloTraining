from ultralytics import YOLO
import os

def detect_fashion_items(image_path, model_path="YOLO12LFashionpedia100Epochs.pt", excluded_labels=None):
    # Load the YOLO model
    model = YOLO(model_path)

    # Set default excluded labels if not provided
    if excluded_labels is None:
        excluded_labels = [
            "neckline", "sleeve", "pocket", "wallet", "lapel",
            "epaulette", "buckle", "zipper", "applique", "bead",
            "bow", "flower", "fringe", "ribbon", "rivet",
            "ruffle", "sequin", "tassel", "collar"
        ]

    # Run the prediction
    results = model.predict(source=image_path, conf=0.2)

    # Extract and filter detections
    detections = []
    for box in results[0].boxes:
        label_id = int(box.cls.item())
        label = model.names[label_id]
        confidence = float(box.conf.item())
        bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

        # Exclude unwanted labels
        if label.lower() in excluded_labels:
            continue

        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": bbox
        })

    return detections