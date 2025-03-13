# Preprocessing script

import cv2
import os
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
DATASET_DIR = "../dataset/"
PROCESSED_DIR = "../dataset/processed/"

# Create processed directory if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_images(folder):
    """Load images from a folder, resize them, and convert to arrays."""
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
            labels.append(0)  # Placeholder label (replace with actual labels)

    return np.array(images), np.array(labels)

def preprocess_and_save():
    """Load, preprocess, and save dataset."""
    print("Loading images...")
    images, labels = load_images(DATASET_DIR)

    # Normalize images
    images = images / 255.0  

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Save preprocessed data
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("✅ Preprocessing complete! Data saved in:", PROCESSED_DIR)

if __name__ == "__main__":
    preprocess_and_save()
