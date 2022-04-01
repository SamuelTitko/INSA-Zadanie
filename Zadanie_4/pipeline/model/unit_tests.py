import numpy as np
import pandas as pd

import config, predict
from pipeline import pipeline, __version__


def test_config():
  import config
  assert 'Age' in config.NUMERICAL_FEATURES
  assert 'Cabin' in config.CATEGORICAL_FEATURES
  assert 'Pclass' in config.CATEGORICAL_FEATURES
  assert 'Sex' in config.CATEGORICAL_FEATURES
  assert 'Parch' in config.CATEGORICAL_FEATURES
  assert 'SibSp' in config.CATEGORICAL_FEATURES
  assert 'Survived' in config.TARGET
  assert 'Age' in config.FEATURES_WITH_NA
  assert 'Cabin' in config.FEATURES_WITH_NA


def test_single_prediction():
  import numpy as np
  import pandas as pd
  import config, predict

  dataset = pd.read_json(config.PATH_TO_TEST_DATASET)
  output = predict.run(dataset[0:1])
  y_pred = output['predictions']

  assert y_pred is not None
  assert isinstance(y_pred[0], np.int64)
  assert y_pred[0] == 0


def test_multiple_predictions():
  import pandas as pd
  import config, predict

  dataset = pd.read_json(config.PATH_TO_TEST_DATASET)
  output = predict.run(dataset)
  y_pred = output['predictions']

  assert y_pred is not None
  assert len(y_pred) <= dataset.shape[0]


def test_shape():
  import pandas as pd
  import config
  from pipeline import pipeline

  dataset = pd.read_csv(config.PATH_TO_TRAIN_DATASET)
  x_train, y_train = dataset[config.FEATURES], dataset[config.TARGET]
  pipeline.fit(dataset[config.FEATURES], dataset[config.TARGET])

  test_shapes = [6, 28, 28]

  for step, test_shape in zip(pipeline.steps[:3], test_shapes):
    dataset = step[1].transform(dataset)
    assert dataset.shape[1] == test_shape


def test_drop_unnecessary_columns():
  import pandas as pd
  import config
  from pipeline import pipeline

  dataset = pd.read_csv(config.PATH_TO_TRAIN_DATASET)
  x_train, y_train = dataset[config.FEATURES], dataset[config.TARGET]
  pipeline.fit(dataset[config.FEATURES], dataset[config.TARGET])

  dataset = pipeline.steps[0][1].transform(dataset)
  dataset.drop(config.FEATURES, axis=1, inplace=True)
  assert dataset.shape[1] == 0


def test_min_max_values():
  import pandas as pd
  import config
  from pipeline import pipeline
  dataset = pd.read_csv(config.PATH_TO_TRAIN_DATASET)
  x_train, y_train = dataset[config.FEATURES], dataset[config.TARGET]
  pipeline.fit(dataset[config.FEATURES], dataset[config.TARGET])

  for step in pipeline.steps[:3]:
    dataset = step[1].transform(dataset)

  assert not (dataset.to_numpy() < 0.0).any()
  assert not (dataset.to_numpy() > 1.0).any()


def test_validation():
  import pandas as pd
  import config, validation
  dataset = pd.read_csv(config.PATH_TO_TRAIN_DATASET)
  dataset = validation.validate_dataset(dataset)
  assert not dataset.isna().sum().any()
