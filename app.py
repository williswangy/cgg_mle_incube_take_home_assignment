from flask import Flask, session, jsonify, request
import json
from ImagePreprocessor import load_train_val_data
from ImagePreprocessor import load_test_data
import logging
from SatelliteImageFetcher import fetch_images_from_geojson
from train import build_model, save_model
import numpy as np
from tensorflow.keras.optimizers import Adam
from evaluate import score_model





app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json','r') as f:
    config = json.load(f)

TRAIN_IMAGE_PATH = config['TRAIN_IMAGE_PATH']
TEST_IMAGE_PATH = config['TEST_IMAGE_PATH']
SIZE = tuple(config['SIZE'])
STAC_URL = config['STAC_URL']
COLLECTION_NAME = config['COLLECTION_NAME']
TIME_OF_INTEREST = config['TIME_OF_INTEREST']
TARGET_WIDTH = config['TARGET_WIDTH']
TRAIN_GEOJSON_PATH = config['TRAIN_GEOJSON_PATH']
TEST_GEOJSON_PATH = config['TEST_GEOJSON_PATH']
ingested_files = []

prediction_model = None


# Global variables to store data
X_train, X_val, y_train, y_val, X_test, y_test = None, None, None, None, None, None

@app.route("/fetch-train-images", methods=['GET'])
def fetch_train_images():
    """
    Fetch and save satellite images for training.

    :return: JSON response with success or error message.
    """
    try:
        train_images = fetch_images_from_geojson(TRAIN_GEOJSON_PATH, save_folder='train_images')
        logger.info(f"Processed {len(train_images)} images from {TRAIN_GEOJSON_PATH}.")
        return jsonify({"status": "success", "message": f"Processed {len(train_images)} images from train.geojson!"})
    except Exception as e:
        logger.error(f"Error fetching train images: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/fetch-test-images", methods=['GET'])
def fetch_test_images():
    """
    Fetch and save satellite images for testing.

    :return: JSON response with success or error message.
    """

    try:
        test_images = fetch_images_from_geojson(TEST_GEOJSON_PATH, save_folder='test_images')
        logger.info(f"Processed {len(test_images)} images from {TEST_GEOJSON_PATH}.")
        return jsonify({"status": "success", "message": f"Processed {len(test_images)} images from test.geojson!"})
    except Exception as e:
        logger.error(f"Error fetching test images: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/load-data", methods=['GET'])
def load_data():
    """
    Load training and testing data for the model.

    :return: JSON response with success or error message.
    """

    global X_train, X_val, y_train, y_val, X_test, y_test
    try:
        X_train, X_val, y_train, y_val = load_train_val_data(TRAIN_IMAGE_PATH)
        X_test, y_test = load_test_data(TEST_IMAGE_PATH)
        return jsonify({"status": "success", "message": "Data loaded successfully!"})
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/predict-score", methods=['GET'])
def predict_score():
    """
    Evaluate the trained model on the test set and provide metrics.

    :return: JSON response with success or error message and evaluation metrics.
    """
    try:
        # Perform model evaluation
        accuracy, f1, confusion = score_model(base_path=TEST_IMAGE_PATH)

        # Return metrics as a JSON response
        result = {
            "status": "success",
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": confusion.tolist()  # Convert numpy array to list
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)