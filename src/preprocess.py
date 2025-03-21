import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
CATEGORIES = ["covid", "pneumonia", "asthma"]

def load_data(img_size=150):
    """Load and preprocess image data."""
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(label)

    data = np.array(data) / 255.0
    data = data.reshape(-1, img_size, img_size, 1)
    labels = np.array(labels)

    return train_test_split(data, labels, test_size=0.2, random_state=42)
