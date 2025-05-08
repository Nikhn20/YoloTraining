import os
import sys
import torch
from sam2_repo.sam2.build_sam import build_sam2
from sam2_repo.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================
# Paths and Model Initialization
# ============================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sam2_repo_path = os.path.abspath("D:/anaconda3/envs/pythonProject8/sam2_repo")
sys.path.append(sam2_repo_path)

sam2_checkpoint = "D:/anaconda3/envs/pythonProject8/sam2_repo/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Initialize SAM2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
mask_generator = SAM2AutomaticMaskGenerator(
     sam2,
     points_per_side=24,
     points_per_batch=32,
     pred_iou_thresh=0.7,
     stability_score_thresh=0.80,
     stability_score_offset=0.7,
     crop_n_layers=1,
     box_nms_thresh=0.7, )



def crop_largest_person(image):


    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Generate masks
    masks = mask_generator.generate(image_np)
    if not masks:
        return None

    # Sort masks by area and select the largest
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    largest_mask_data = masks_sorted[0]
    largest_mask = largest_mask_data['segmentation']

    # Get bounding box coordinates
    y_indices, x_indices = np.where(largest_mask)
    if y_indices.size == 0 or x_indices.size == 0:
        return None

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Crop the image and mask to the bounding box
    cropped_image = image_np[y_min:y_max, x_min:x_max]
    cropped_mask = largest_mask[y_min:y_max, x_min:x_max]

    # Apply the mask to black out the background
    cropped_masked = np.zeros_like(cropped_image)
    cropped_masked[cropped_mask] = cropped_image[cropped_mask]

    return cropped_masked

img = Image.open("detecthoomans/person_76.jpg").convert("RGB")
cropped_masked = crop_largest_person(img)

output_img = Image.fromarray(cropped_masked)
output_img.save("croppy.png")