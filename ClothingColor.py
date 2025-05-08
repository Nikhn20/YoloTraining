import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class ClothingColorAnalyzer:
    def __init__(self):
        # Initialize model and processor
        self.model_name = "mattmdjaga/segformer_b2_clothes"
        self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.eval()

        # Class definitions
        self.label_names = {
            0: "Background",
            1: "Hat",
            2: "Hair",
            3: "Sunglasses",
            4: "Upper-clothes",
            5: "Skirt",
            6: "Pants",
            7: "Dress",
            8: "Belt",
            9: "Left-shoe",
            10: "Right-shoe",
            11: "Face",
            12: "Left-leg",
            13: "Right-leg",
            14: "Left-arm",
            15: "Right-arm",
            16: "Bag",
            17: "Scarf"
        }

    def _compute_segmentation_map(self, image):
        """Internal method to compute segmentation map"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        seg_map = torch.argmax(logits, dim=1)

        # Resize to original dimensions
        orig_size = image.size[::-1]  # (height, width)
        seg_map = F.interpolate(
            seg_map.unsqueeze(0).float(),
            size=orig_size,
            mode="nearest"
        )[0, 0].cpu().numpy()

        return seg_map

    def _rgb_to_hex(self, rgb):
        """Convert RGB tuple to hexadecimal color code"""
        return "#{:02x}{:02x}{:02x}".format(*map(int, rgb))

    def analyze_clothing_colors(self, image):
        """
        Analyze an image and return clothing items with average colors.

        Args:
            image (PIL.Image.Image/np.ndarray): Input image in RGB format

        Returns:
            list: List of dictionaries with clothing items and their colors
                  Format: [{"label": str, "rgb": tuple, "hex": str}, ...]
        """
        # Ensure image is in RGB format
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # Assume BGR if from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.convert("RGB")

        # Compute segmentation
        seg_map = self._compute_segmentation_map(pil_image)
        image_np = np.array(pil_image)

        # Calculate mean colors
        results = []
        for cls in np.unique(seg_map):
            if cls == 0:  # Skip background
                continue

            mask = seg_map == cls
            if np.sum(mask) == 0:
                continue

            # Calculate mean color
            mean_color = (
                np.mean(image_np[:, :, 0][mask]),
                np.mean(image_np[:, :, 1][mask]),
                np.mean(image_np[:, :, 2][mask])
            )

            label = self.label_names.get(cls, f"Class {cls}")
            results.append({
                "label": label,
                "rgb": tuple(np.round(mean_color).astype(int)),
                "hex": self._rgb_to_hex(mean_color)
            })

        return results