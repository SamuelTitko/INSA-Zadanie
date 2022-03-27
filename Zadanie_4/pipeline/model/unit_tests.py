import numpy as np
import pandas as pd

import config, predict


def test_single_prediction():
  dataset = pd.read_json(config.PATH_TO_TEST_DATASET)
  output = predict.run(dataset[0:1])
  y_pred = output['predictions']

  assert y_pred is not None
  assert isinstance(y_pred[0], np.int64)
  assert y_pred[0] == 0


def test_multiple_predictions():
  dataset = pd.read_json(config.PATH_TO_TEST_DATASET)
  output = predict.run(dataset)
  y_pred = output['predictions']

  assert y_pred is not None
  assert len(y_pred) <= dataset.shape[0]
