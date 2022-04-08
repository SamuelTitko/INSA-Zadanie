import logging
from sklearn.pipeline import Pipeline

from model import config, manager, transformers

logger = logging.getLogger(__name__)
__version__ = manager.get_version(config.PATH_TO_VERSION)

pipeline = Pipeline([
  ('filter_transformer', transformers.FilterTransformer(config.FEATURES)),
  ('categorical_transformer', transformers.CategoricalTransformer(config.CATEGORICAL_FEATURES)),
  ('numerical_transformer', transformers.NumericalTransformer(config.NUMERICAL_FEATURES)),
  ("classifier_model", config.CLASSIFIER),
])