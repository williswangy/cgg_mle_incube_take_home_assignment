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
    """
    Loads and normalizes images from a specified folder and associates them with a label.

    Args:
        folder (str): The path to the folder containing images.
        label (int): The label associated with the images in the specified folder.
        size (tuple, optional): The target size to which the images should be resized. Defaults to `SIZE`.

    Returns:
        List[Tuple[np.array, int]]: A list of tuples where each tuple contains the normalized image array and its associated label.
    """
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
    """
    Retrieves a dataset of images and their associated labels from a base path.

    Args:
        base_path (str): The base path from which images are to be loaded.
        size (tuple, optional): The target size to which the images should be resized. Defaults to `SIZE`.

    Returns:
        Tuple[List[np.array], List[int]]: Two lists where the first list contains image arrays and the second list contains their associated labels.
    """
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

def load_train_val_data(base_path):
    """
    Loads and splits the dataset into training and validation subsets.

    Args:
        base_path (str): The base path from which images are to be loaded.

    Returns:
        Tuple[List[np.array], List[np.array], List[int], List[int]]:
        Returns the training images, validation images, training labels, and validation labels.
    """
    logging.info("Starting data loading process for training and validation datasets...")
    X, y = get_dataset(base_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training dataset split into {len(X_train)} training samples and {len(X_val)} validation samples.")
    return X_train, X_val, y_train, y_val

def load_test_data(base_path):
    """
    Loads the test dataset.

    Args:
        base_path (str): The base path from which images are to be loaded.

    Returns:
        Tuple[List[np.array], List[int]]: Returns the test images and their associated labels.
    """
    logging.info("Starting data loading process for test dataset...")
    X_test, y_test = get_dataset(base_path)
    logging.info(f"Loaded {len(X_test)} test samples.")
    return X_test, y_test


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_train_val_data(TRAIN_IMAGE_PATH)
    X_test, y_test = load_test_data(TEST_IMAGE_PATH)
