import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # 1. Load an image
    image_path = "D:/anaconda3/envs/pythonProject8/person_244.jpg"
    image = Image.open(image_path).convert("RGB")

    # 2. Load YOLOS processor & model
    image_processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    model.eval()

    # 3. Prepare inputs
    inputs = image_processor(images=image, return_tensors="pt")

    # 4. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Post-process
    w, h = image.size
    target_sizes = torch.tensor([[h, w]], dtype=torch.long)  # shape: (1, 2)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.4,
        target_sizes=target_sizes
    )

    detections = results[0]
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    print("Available labels:", list(model.config.id2label.values()))
    # 5.5 Filter out unwanted labels
    excluded_labels = ["neckline", "sleeve", "pocket", "wallet", "lapel", "epaulette", "buckle", "zipper", "applique",
                       "bead", "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel", "collar"]  # Add/remove labels here
    keep_indices = [
        i for i, label_id in enumerate(labels)
        if model.config.id2label[label_id.item()].lower() not in excluded_labels
    ]

    # Apply the filter
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # 6. Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    color_map = plt.cm.get_cmap("hsv", len(boxes))

    for i, (box, score, label_id) in enumerate(zip(boxes, scores, labels)):
        # box is (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = box.detach().cpu().tolist()
        width = x_max - x_min
        height = y_max - y_min

        # Assign a color from the palette for each bounding box
        color = color_map(i)  # color_map(i) returns an RGBA tuple

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        class_name = model.config.id2label[label_id.item()]
        ax.text(
            x_min,
            y_min,
            f"{class_name}: {score:.2f}",
            fontsize=12,
            color="white",
            bbox=dict(facecolor=color, alpha=0.5, pad=2)
        )

    plt.axis("off")
    plt.title("YOLOS Fashionpedia Detection (Colored Boxes)")
    plt.show()

if __name__ == "__main__":
    main()