import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import YolosImageProcessor, YolosForObjectDetection

# 1. Load model and processor
processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
model.eval()

# 2. Paths
val_images_dir = r'/val_test2020/test'
val_annotations = r'C:\Users\mnj-7\PycharmProjects\YoloTraining\val_test2020\instances_attributes_val2020.json'

# 3. Load Fashionpedia validation annotations
coco = COCO(val_annotations)

# 4. Prepare predictions
coco_results = []

for img_id in tqdm(coco.getImgIds()):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(val_images_dir, img_info['file_name'])

    # Load and preprocess the image
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])  # height, width
    results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    for box, score, label in zip(boxes, scores, labels):
        # YOLOS outputs boxes as (x0, y0, x1, y1)
        x_min, y_min, x_max, y_max = box.tolist()
        width = x_max - x_min
        height = y_max - y_min

        coco_results.append({
            "image_id": img_id,
            "category_id": int(label.item()) + 1,  # Fashionpedia labels start at 1
            "bbox": [x_min, y_min, width, height],
            "score": float(score.item())
        })

# 5. Save results
with open("yolos_fashionpedia_val_predictions.json", "w") as f:
    json.dump(coco_results, f)

# 6. Evaluate with COCO API
coco_dt = coco.loadRes("yolos_fashionpedia_val_predictions.json")
coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()