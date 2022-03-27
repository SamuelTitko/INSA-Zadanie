import logging
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config
from src import manager, validation

logger = logging.getLogger(__name__)

__version__ = manager.get_version(config.PATH_TO_VERSION)


def run():
  logger.info('Running test.py file.')
  logger.info(f'Loading dataset from: {config.PATH_TO_TEST_DATASET}')
  dataset = pd.read_json(config.PATH_TO_TEST_DATASET)
  dataset = validation.validate_dataset(dataset)

  pipeline = manager.load_pipeline(
    config.PATH_TO_MODEL,
    config.MODEL_NAME,
    __version__
  )
  score = pipeline.score(dataset[config.FEATURES], dataset[config.TARGET])
  y_pred = pipeline.predict(dataset[config.FEATURES])
  f1score = f1_score(dataset[config.TARGET], y_pred)
  matrix = confusion_matrix(dataset[config.TARGET], y_pred)

  logger.info(f'Using model version: {__version__}')
  logger.info(f'Score: {score*100:.2f}%')
  logger.info(f'F1 Score: {f1score*100:.2f}%')
  logger.info(f'Confusion matrix:\n{matrix}')
  logger.info(f'Done!')


if __name__ == '__main__':
  run()