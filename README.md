# 3D Human Model Generator – Python Backend

This repository contains the Python backend for a 3D human model generation system. It includes person detection, clothing color recognition, hair type classification, and integration with a Unity frontend.

## 📁 Project Structure

```
YoloTraining/
├── .idea/                      
├── FolderTiViz/                
├── LongShortHairModels/        # Hair classification models
├── Unused/                     # Archived or unused scripts
├── ValidationScripts/          # Model validation scripts
├── testdata/                   
├── testdata2/                  
├── BilledeMed3Personer.PNG     
├── ClothingColor.py            # Detects mean clothing colors
├── FindHairType.py             # Identifies hair type (long/short)
├── FindPeople.py               # Detects people in images
├── NewYOLO100ClothingFinder.py # Clothing detection using the trained model
├── YOLO12LFashionpedia100Epochs.pt  # YOLO model trained on Fashionpedia
├── YOLO12M25Epochs.pt          
├── yolov8n.pt                  # Pretrained YOLOv8n model
├── new_image.jpg               
├── new_image3.jpg              
├── server.py                   # Python backend server script
├── .gitattributes
├── .gitignore
└── README.md                   # Project documentation
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Nikhn20/YoloTraining.git
cd YoloTraining
```

### 2. Install Dependencies

Use `pip` to install the required Python libraries:

```bash
pip install opencv-python matplotlib torch numpy Pillow transformers ultralytics
```

> **Note:** Make sure you have Python 3.8 or above installed.

### 3. Run the Backend Server

```bash
python server.py
```

This will start the backend server which processes image data for use in the Unity frontend.

### 4. Launch Unity Application

Once the backend is running, start the Unity application to connect and visualize the 3D human model.

## Features

- **YOLOv8 Detection**: Person and clothing detection using pretrained and custom YOLO models.
- **Clothing Color Analysis**: Extracts dominant clothing colors for model customization.
- **Hair Type Detection**: Classifies individuals by hair type (e.g., long or short).
- **Unity Integration**: Communicates with Unity frontend to dynamically generate 3D models.

## Notes

- The `Unused/` folder contains scripts that are not actively used.
- Ensure Unity and Python can communicate on the same network or localhost.
