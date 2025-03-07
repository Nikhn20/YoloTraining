import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def prepare_fashion_mnist(root_dir="Fashion-MNIST", val_size=0.2):
    """
    Downloads the Fashion MNIST dataset, splits into train/val, 
    and creates YOLO-style image and label folders.
    """

    # 1. Download dataset with torchvision
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=root_dir, train=True, download=True, transform=transform)
    test_data  = datasets.FashionMNIST(root=root_dir, train=False, download=True, transform=transform)

    # Combine train+test to manually split
    all_data = []
    for img, label in train_data:
        all_data.append((np.array(img[0]), label))  # img is a tensor: shape [1,28,28]

    for img, label in test_data:
        all_data.append((np.array(img[0]), label))

    # Convert list to arrays
    images = np.array([item[0] for item in all_data])
    labels = np.array([item[1] for item in all_data])

    # 2. Train-Val split (80/20)
    train_idx, val_idx = train_test_split(range(len(images)), test_size=val_size, stratify=labels, random_state=42)

    # 3. Create directories
    img_train_dir = os.path.join(root_dir, 'images', 'train')
    img_val_dir   = os.path.join(root_dir, 'images', 'val')
    lbl_train_dir = os.path.join(root_dir, 'labels', 'train')
    lbl_val_dir   = os.path.join(root_dir, 'labels', 'val')

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    # 4. Function to save images and labels
    def save_image_label(idx_list, split='train'):
        if split == 'train':
            img_dir = img_train_dir
            lbl_dir = lbl_train_dir
        else:
            img_dir = img_val_dir
            lbl_dir = lbl_val_dir

        for i in idx_list:
            img_array = images[i]
            label = labels[i]

            # Save image as PNG
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))  # if not normalized
            img_filename = f"{i}.png"
            img_path = os.path.join(img_dir, img_filename)
            img_pil.save(img_path)

            # YOLO bounding box for entire image:
            # Normalized center_x, center_y, width, height
            # Since image size is 28x28, bounding box covers entire image:
            x_center = 0.5
            y_center = 0.5
            width = 1.0
            height = 1.0

            # YOLO format: class x_center y_center width height
            yolo_label_str = f"{label} {x_center} {y_center} {width} {height}\n"

            # Save label
            label_filename = f"{i}.txt"
            label_path = os.path.join(lbl_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write(yolo_label_str)

    # 5. Write out train set
    save_image_label(train_idx, split='train')
    # 6. Write out val set
    save_image_label(val_idx, split='val')

    print("Fashion MNIST prepared in YOLO format!")

if __name__ == "__main__":
    prepare_fashion_mnist()
