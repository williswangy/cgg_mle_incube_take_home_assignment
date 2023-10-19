

## Machine Learning Workflow Components

### `SatelliteImageFetcher.py`
Automates the collection of satellite imagery data for specific geographic regions (AOIs). It fetches the least cloudy images from a SpatioTemporal Asset Catalog (STAC) API, resizes them to match the AOIs, and saves them with associated details and labels. This script is structured with functions for data retrieval, processing, and storage. It processes two GeoJSON files to obtain training and testing datasets, making it valuable for machine learning and remote sensing applications.

### `ImagePreprocessor.py`
Designed for loading and preprocessing image datasets for machine learning tasks. It reads and normalizes images from specified folders, associates them with labels, and splits the data into training and validation sets using Scikit-Learn's `train_test_split` function. Additionally, it loads a separate test dataset. Configuration parameters are read from a `config.json` file, making it adaptable to different datasets and image sizes. This script streamlines the preparation of image data for model training and evaluation.

### `train.py`
Trains a binary classification model using the VGG16 architecture with TensorFlow and Keras. It starts by setting up logging and loading configuration parameters from a `config.json` file. The `build_model` function extends VGG16 with additional layers for binary classification. The script loads training and validation data, preprocesses it, and compiles the model with a specified optimizer, loss function, and evaluation metrics. After training, the model is saved to disk. This script provides an efficient workflow for training and saving binary classification models for image data.

### `evaluate.py`
Evaluates a pre-trained binary classification model using test data and saves the evaluation metrics. It loads a pre-trained Keras model from an H5 file and then loads and preprocesses the test dataset. The script predicts labels for the test data and calculates key metrics such as accuracy, F1 score, and a confusion matrix to assess the model's performance. These metrics are then saved to a text file. This script streamlines the assessment of a trained model's performance on new data and records the evaluation results for further analysis.

### `app.py`
A Flask application serving as an API for a machine learning workflow. It provides endpoints for tasks like fetching and processing satellite images for training and testing, loading and preparing data, and evaluating a pre-trained machine learning model on the test dataset. Users can interact with these tasks via HTTP requests. The application uses global variables to store data and leverages external modules for data preprocessing, model building, and evaluation. Overall, it simplifies working with geospatial data and machine learning models through a user-friendly API.

### `apicalls.py`
Automates the interaction with a locally hosted Flask API by making sequential HTTP GET requests to various endpoints. It fetches, processes, and loads satellite images for training and testing datasets and evaluates a pre-trained model on the test dataset. Responses are structured in a dictionary and saved as a JSON file for documentation and further analysis. This script offers a streamlined way to execute and document a machine learning workflow using API endpoints and configuration parameters.


## Running the Flask App

1. Navigate to the base folder `cgg_mle_incube_take_home_assignment/cgg_mle_incube_take_home_assignment/` in your terminal.

2. Open one terminal window.

3. Run the Flask app by executing the following command:

   ```
   python app.py
   ```

   This will start the Flask application, and it will be ready to receive HTTP requests.

4. Open another terminal window.

5. Run the `apicalls.py` script by executing the following command:

   ```
   python apicalls.py
   ```

   This script automates the interaction with the Flask API by making HTTP GET requests to various endpoints.

6. The entire process, including API calls and responses, will be tracked in the `api_responses.txt` file located in the same folder:

   ```
   cgg_mle_incube_take_home_assignment/cgg_mle_incube_take_home_assignment/api_responses.txt
   ```

   This file will contain details of all the API requests and their corresponding responses, providing a record of the entire workflow.

Please follow these steps to run the Flask app and document the process using the `api_responses.txt` file.



## Production Folder Contents

### `ingestedfiles.txt`
This file contains information about the ingestion process of the train and test images. It likely includes details about which satellite images were fetched, resized, and used for training and testing datasets. It serves as a record of the data ingestion workflow.

### `latestscores.txt`
The `latestscores.txt` file stores metrics results from model evaluation, including accuracy, F1 score, and a confusion matrix. These metrics help assess the performance of the machine learning model on the test dataset. Reviewing this file provides insights into how well the model is performing on new, unseen data.

### `vgg16_model`
This file represents a saved trained model using the VGG16 architecture. Saved models like these are valuable for quicker prediction on new data without the need for retraining. The model can be loaded and used to make predictions on satellite images efficiently.

These files in the production folder are essential for tracking and managing the machine learning workflow's progress and performance, ensuring that the model is up to date and ready for deployment.