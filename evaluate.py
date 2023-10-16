import os
import json
import numpy as np
import logging
from sklearn import metrics
from tensorflow.keras.models import load_model
from ImagePreprocessor import load_test_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json', 'r') as file:
    config = json.load(file)


def load_h5_model(model_path: str):
    """
    Loads a trained H5 model from disk.

    Args:
        model_path (str): The path to the H5 model file on disk.

    Returns:
        Model: A loaded Keras model.

    Raises:
        Exception: If there is an issue loading the model.
    """
    try:
        model = load_model(model_path)
        logging.info("H5 Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error occurred while loading H5 model: {e}")
        raise


def score_model():
    """
    Loads a trained model, evaluates it on the test set, and saves evaluation metrics to a text file.

    Returns:
        Tuple[float, float, np.array]:
        Returns the accuracy, F1 score, and confusion matrix for the model on the test data.
    """

    # Load trained model
    model = load_h5_model(config['MODEL']['SAVE_MODEL'])

    # Load and preprocess test dataset
    X_test, y_test = load_test_data()
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Predict the labels for test set
    y_pred = model.predict(X_test)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]  # Threshold the predictions for binary classification

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    confusion = metrics.confusion_matrix(y_test, y_pred)

    # Save metrics to a txt file
    metrics_str = (
        f"Accuracy        - {accuracy:.4f}\n"
        f"F1 Score        - {f1:.4f}\n"
        f"Confusion Matrix:\n"
        f"{confusion[0][0]:8d} {confusion[0][1]:8d}\n"
        f"{confusion[1][0]:8d} {confusion[1][1]:8d}\n"
    )

    scorespath = os.path.join(config['SCORE_PATH'], 'latestscore.txt')
    with open(scorespath, 'w') as f:
        f.write(metrics_str)

    logging.info(f"Metrics computed and saved to {scorespath}")
    return accuracy, f1, confusion


if __name__ == "__main__":
    score_model()
