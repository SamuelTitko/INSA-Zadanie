import logging
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from src import manager, transformers

logger = logging.getLogger(__name__)

__version__ = manager.get_version(config.PATH_TO_VERSION)


def run():
  logger.info('Running train.py file.')
  logger.info(f'Loading dataset from: {config.PATH_TO_TRAIN_DATASET}')
  dataset = pd.read_csv(config.PATH_TO_TRAIN_DATASET)
  x_train, y_train = dataset[config.FEATURES], dataset[config.TARGET]

  logger.info(f'Creating new pipeline.')
  pipeline = Pipeline([
    ['categorical transformer', transformers.CategoricalTransformer(config.CATEGORICAL_FEATURES)],
    ['numerical transformer', transformers.NumericalTransformer(config.NUMERICAL_FEATURES)],
    ["classifier model", config.CLASSIFIER],
  ])

  logger.info(f'Training pipeline...')
  pipeline.fit(x_train, y_train)

  manager.save_pipeline(config.PATH_TO_MODEL, config.MODEL_NAME, __version__, pipeline)
  logger.info(f'Done!')


if __name__ == "__main__":
  run()
