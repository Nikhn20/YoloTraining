from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Load your trained model (.pt file)
    model = YOLO(R"/YOLO12M25Epochs.pt")  # <-- replace with the path to your model weights

    # Validate on your dataset (defined by your data.yaml)
    results = model.val(data=R'C:\Users\mnj-7\PycharmProjects\YoloTraining\Fashionpedia_yolo\data.yaml')

    # Print validation results
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Per-class precision: {results.box.p}")
    print(f"Per-class recall: {results.box.r}")

    # Print inference speed info
    print(f"\nSpeed (inference) per image: {results.speed['inference']:.2f} ms")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
