from ultralytics import YOLO

# Load your trained model
model = YOLO("/YOLO12M25Epochs.pt")  # path to your model weights

 #Predict on a folder of images or a single image
results = model.predict(source=r"C:\Users\mnj-7\PycharmProjects\YoloTraining\FolderTilViz\0d5d5e00e589fe5641b919179794994b.jpg",  # can be a folder or single file
                      conf=0.20,  # confidence threshold, adjust as needed
                        save=False)  # save visualized predictions (bounding boxes) to runs/predict

#img = Image.open("FolderTilTestData/person_13.jpg")
#detections = detect_fashion_items(img)

#for detection in detections:
#    print(detection)


#detectedcc = find_clothes("FolderTilTestData/person_13.jpg", confidence_threshold=0.2)

#img = Image.open("FolderTilTestData/person_13.jpg")
#detections = find_clothes(img)
#print(detections)