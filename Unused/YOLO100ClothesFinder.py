from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the model
model = YOLO("/YOLO12LFashionpedia100Epochs.pt")  # Your custom weights

def find_clothes(img, confidence_threshold=0.2, excluded_labels=None):
    """
    Detect clothing items in an image using YOLO.
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Convert PIL to numpy for YOLO inference
    img_np = np.array(img)

    # Run inference
    results = model(img_np)[0]


    # Default excluded labels
    if excluded_labels is None:
        excluded_labels = ["neckline", "sleeve", "pocket", "wallet", "lapel",
                           "epaulette", "buckle", "zipper", "applique", "bead",
                           "bow", "flower", "fringe", "ribbon", "rivet",
                           "ruffle", "sequin", "tassel", "collar"]

    # Filter detections
    detections = []
    for box in results.boxes:
        label_id = int(box.cls.item())
        label = model.names[label_id]
        if label.lower() in excluded_labels:
            continue
        detections.append({
            "label": label,
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
        })


    return detections
