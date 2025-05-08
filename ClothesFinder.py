import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import numpy as np

# Initialize model and processor once (not inside the function)
MODEL_NAME = "valentinafeve/yolos-fashionpedia"
image_processor = YolosImageProcessor.from_pretrained(MODEL_NAME)
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
model.eval()


def find_clothes(img, confidence_threshold=0.4, excluded_labels=None):
    """
    Detect clothing items in an image.

    Args:
        img (PIL.Image/ndarray): Input image (RGB format)
        confidence_threshold (float): Minimum detection confidence (0-1)
        excluded_labels (list): Labels to exclude from results

    Returns:
        list: List of dictionaries with detection results
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Default excluded labels
    if excluded_labels is None:
        excluded_labels = ["neckline", "sleeve", "pocket", "wallet", "lapel",
                           "epaulette", "buckle", "zipper", "applique", "bead",
                           "bow", "flower", "fringe", "ribbon", "rivet",
                           "ruffle", "sequin", "tassel", "collar"]

    # Process image
    inputs = image_processor(images=img, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    w, h = img.size
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=confidence_threshold,
        target_sizes=torch.tensor([[h, w]])
    )[0]

    # Filter detections
    detections = []
    for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
        label = model.config.id2label[label_id.item()]

        if label.lower() in excluded_labels:
            continue

        detections.append({
            "label": label,
            "confidence": score.item(),
            "bbox": box.tolist()  # [x_min, y_min, x_max, y_max]
        })

    return detections