import logging
import config

logger = logging.getLogger(__name__)

def validate_dataset(dataset):
  dataset = dataset[config.FEATURES + config.TARGET]
  dataset = dataset.dropna(subset=config.FEATURES_WITH_NA)
  logger.info('Dataset has been validated!')
  return dataset