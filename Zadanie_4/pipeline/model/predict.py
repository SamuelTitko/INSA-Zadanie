import pandas as pd
import numpy as np
import logging

import config
from src import manager, validation

logger = logging.getLogger(__name__)

__version__ = manager.get_version(config.PATH_TO_VERSION)


def run(dataset):
  logger.info('Running predict.py file.')
  dataset = validation.validate_dataset(dataset)

  pipeline = manager.load_pipeline(
    config.PATH_TO_MODEL,
    config.MODEL_NAME,
    __version__
  )
  prediction = pipeline.predict(dataset[config.FEATURES])
  response = {'predictions': list(prediction)}

  logger.info(f'Using model version: {__version__}')
  logger.info(f'Input dataset:\n{dataset}\n')
  logger.info(f'Predictions:\n{response}')
  logger.info(f'Done!')

  return response


if __name__ == '__main__':
  run(pd.read_json(config.PATH_TO_TEST_DATASET))