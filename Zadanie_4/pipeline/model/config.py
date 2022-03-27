import pathlib
import logging
from sklearn.ensemble import RandomForestClassifier

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
PATH_TO_VERSION = PACKAGE_ROOT / 'VERSION'
PATH_TO_FULL_DATASET = PACKAGE_ROOT / 'dataset/dataset.csv'
PATH_TO_TRAIN_DATASET = PACKAGE_ROOT / 'dataset/train.csv'
PATH_TO_TEST_DATASET = PACKAGE_ROOT / 'dataset/test.json'
PATH_TO_MODEL = PACKAGE_ROOT / 'trained_models'

NUMERICAL_FEATURES = ['Age']
CATEGORICAL_FEATURES = ['Cabin', 'Pclass','Sex', 'Parch', 'SibSp']
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = ['Survived']

FEATURES_WITH_NA = ['Age', 'Cabin']

CLASSIFIER = RandomForestClassifier()
MODEL_NAME = 'model'

PATH_TO_LOG = PACKAGE_ROOT / 'output.log'

logging.basicConfig(
  filename=PATH_TO_LOG,
  filemode='a',
  format='%(asctime)s | %(levelname)8s: %(message)s',
  level=logging.DEBUG,
  datefmt='%Y-%m-%d %H:%M:%S'
)
