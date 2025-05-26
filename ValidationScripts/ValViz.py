import os
import cv2
import matplotlib.pyplot as plt
import random

# Paths to validation images and labels
IMAGE_DIR = r'/Fashionpedia_yolo/images'
LABEL_DIR = r'/Fashionpedia_yolo/labels'
OUTPUT_DIR = '../output/'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate random colors for each class (consistent across images)
NUM_CLASSES = 46
random.seed(42)  # For consistent colors across runs
class_colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in
                range(1, NUM_CLASSES + 1)}

# Class names (replace with actual class names if available)
class_names = {i: f'Class_{i}' for i in range(1, NUM_CLASSES + 1)}

# Get all image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Read the image
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Find corresponding label file
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(LABEL_DIR, label_file)
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {image_file}")
        continue

    # Read label file
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_file}: {line.strip()}")
                continue
            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO format to bounding box
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Draw the bounding box
            color = class_colors.get(class_id, (0, 255, 0))  # Use pre-generated color
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Add the class label
            label = class_names.get(class_id, f'Class_{class_id}')
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save the annotated image
    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

print("Processing complete.")