from ultralytics import YOLO
from PIL import Image

from YOLO100ClothesFinder import find_clothes

# Load your trained model
model = YOLO("C:/Users/mnj-7/PycharmProjects/YoloTraining/YOLO12LFashionpedia100Epochs.pt")  # path to your model weights

 #Predict on a folder of images or a single image
results = model.predict(source="FolderTilTestData/person_13.jpg",  # can be a folder or single file
                       conf=0.20,  # confidence threshold, adjust as needed
                        save=True)  # save visualized predictions (bounding boxes) to runs/predict


#detectedcc = find_clothes("FolderTilTestData/person_13.jpg", confidence_threshold=0.2)

#img = Image.open("FolderTilTestData/person_13.jpg")
#detections = find_clothes(img)
#print(detections)