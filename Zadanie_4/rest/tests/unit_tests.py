import requests
from flask import jsonify
from model import manager, config


def test_status():
  response = requests.get('http://127.0.0.1:5000/status')

  assert response.status_code == 200


def test_version():
  response = requests.get('http://127.0.0.1:5000/version')

  assert response.status_code == 200


def test_predictions():
  dataset = manager.load_dataset(config.PATH_TO_TEST_DATASET)
  response = requests.post('http://127.0.0.1:5000/predict', dataset.to_json())
  predictions = response.json()

  assert response.status_code == 200
  assert isinstance(predictions, dict)
  assert 'predictions' in predictions.keys()
  assert dataset.shape[0] >= len(predictions['predictions'])
