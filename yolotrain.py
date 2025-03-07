import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import torch
print("CUDA available?", torch.cuda.is_available())


# If you have a custom model YAML, put it in your working directory or specify the full path
  # or 'yolov8n.pt' if you want to start from a pretrained checkpoint

def main():
    model = YOLO("yolo11m.pt")
    results = model.train(
        data="Fashion-MNIST/data.yaml",
        epochs=5,
        imgsz=28,
        workers=0,
        translate=0.2,
        perspective=0.0005  # or 1
    )


if __name__ == "__main__":
    main()