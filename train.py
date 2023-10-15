import numpy as np
import os
import logging
import json
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from ImagePreprocessor import load_all_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json', 'r') as file:
    config = json.load(file)

def build_model(input_shape):
    """
    Builds a binary classification model based on the VGG16 architecture.

    Args:
        input_shape (tuple): The shape of the input tensor.

    Returns:
        Model: A TensorFlow Keras model with VGG16 base and additional custom layers.
    """
    # Load the VGG16 model with weights pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the VGG16 model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for binary classification
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def save_model(model, filepath):
    """
    Save the trained model to disk using TensorFlow's save method.

    Args:
        model (Model): The TensorFlow Keras model to save.
        filepath (str): The path where the model should be saved.
    """
    logging.info(f"Saving model to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)

if __name__ == "__main__":
    """
    Entry point for the training process. 
    Loads the training and validation data, builds the model, trains it, and saves the trained model to disk.
    """
    logger.info("Starting the training process...")

    # Load the data
    logger.info("Loading data...")
    X_train, X_val, y_train, y_val, _, _ = load_all_data()

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    model = build_model(X_train[0].shape)

    optimizer = Adam(learning_rate=config['MODEL']['LEARNING_RATE'])

    model.compile(
        optimizer=optimizer,
        loss=config['MODEL']['LOSS'],
        metrics=config['MODEL']['METRICS']
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['MODEL']['EPOCHS'],
        batch_size=config['MODEL']['BATCH_SIZE']
    )

    # Save the model
    save_model(model, config['MODEL']['SAVE_PATH'])

    logger.info("Training completed.")
