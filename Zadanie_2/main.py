import config
import pandas as pd
from pipeline import Pipeline

if __name__ == '__main__':
  pipeline = Pipeline(
    config.CLASSIFIER,
    config.NUMERICAL,
    config.CATEGORICAL,
    config.NUMERICAL_FEATURES,
    config.CATEGORICAL_FEATURES,
    config.TARGET,
  )

  dataset = pd.read_csv(PATH_TO_DATASET)

  pipeline.fit(dataset)
  score, matrix = pipeline.evaluate_model()
  print(f'Accuracy: {score*100:.2f}%')
  print(f'Confusion matrix:\n{matrix}')