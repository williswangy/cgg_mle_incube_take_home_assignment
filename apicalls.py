import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

train_image_path = os.path.join(config['TRAIN_IMAGE_PATH'])
test_image_path = os.path.join(config['TEST_IMAGE_PATH'])
base_path = os.path.join(config['BASE_PATH'])

def make_request(endpoint):
    """Helper function to make a request and return a meaningful response"""
    try:
        response = requests.get(URL + endpoint)
        response.raise_for_status()
        return response.content.decode('utf-8')
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"Error occurred: {err}"

# Call each API endpoint and store the responses
responses = {}

# Fetch Train Images
responses['TrainImages'] = make_request('/fetch-train-images')

# Fetch Test Images
responses['TestImages'] = make_request('/fetch-test-images')

# Load Data
responses['LoadData'] = make_request('/load-data')


# Predict Score
responses['PredictScore'] = make_request('/predict-score')

directory_path = os.path.join(base_path, 'cgg_mle_incube_take_home_assignment')
os.makedirs(directory_path, exist_ok=True)
# Modify this to your needs
filepath = os.path.join(directory_path, 'api_responses.txt')
with open(filepath, 'w') as f:
    f.write(json.dumps(responses, indent=4))
if __name__ == "__main__":
    print("API responses saved to:", filepath)
