from ultralytics import YOLO

# Load the model once during initialization
model = YOLO("LongShortHairModels/best.pt")  # Update path if needed


def predict_hair_type(image_source, confidence_level=0.4):
    """
    Predicts hair length from an image source.

    Args:
        image_source (str/np.ndarray/PIL.Image): Input image source
        confidence_level (float): Confidence threshold (0-1)

    Returns:
        str: 'short', 'long', or 'No Hair Found'
    """
    # Run prediction
    results = model.predict(
        source=image_source,
        conf=confidence_level,
        save=False,
        verbose=False
    )

    # Check for valid detections
    if not results or not results[0].boxes:
        return "No Hair Found"

    # Extract boxes and confidence scores
    boxes = results[0].boxes
    confidences = boxes.conf

    # Get index of highest confidence detection
    max_idx = confidences.argmax()
    class_id = int(boxes.cls[max_idx].item())
    class_name = model.names[class_id].lower()

    # Return simplified result
    return 'short' if 'short' in class_name else 'long'