import os
from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json', 'r') as file:
    config = json.load(file)

TRAIN_IMAGE_PATH = config['TRAIN_IMAGE_PATH']
TEST_IMAGE_PATH = config['TEST_IMAGE_PATH']
SIZE = tuple(config['SIZE'])


def load_images_from_folder(folder, label, size=SIZE):
    images = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                img = img.resize(size)
                normalized_img = np.array(img) / 255.0  # Normalize the image
                images.append((normalized_img, label))
        except Exception as e:
            logging.error(f"Error processing file {filename} in {folder}: {e}")
    logging.info(f"Loaded {len(images)} images from {folder} with label {label}")
    return images


def get_dataset(base_path, size=SIZE):
    images = []
    labels = []

    for label in ['0', '1']:
        path = os.path.join(base_path, label)
        data = load_images_from_folder(path, int(label), size)
        for img, lbl in data:
            images.append(img)
            labels.append(lbl)

    logging.info(f"Retrieved dataset from {base_path} with {len(images)} images in total.")
    return images, labels

def load_all_data():
    logging.info("Starting data loading process...")
    X, y = get_dataset(TRAIN_IMAGE_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training dataset split into {len(X_train)} training samples and {len(X_val)} validation samples.")
    X_test, y_test = get_dataset(TEST_IMAGE_PATH)
    logging.info(f"Loaded {len(X_test)} test samples.")
    logging.info("Data loading process completed.")
    return X_train, X_val, y_train, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, X_test, y_test = load_all_data()
