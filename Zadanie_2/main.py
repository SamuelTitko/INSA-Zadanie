import config
import pandas as pd
from pipeline import Pipeline

if __name__ == '__main__':
  pipeline = Pipeline(
    config.CLASSIFIER,
    config.NUMERICAL_FEATURES,
    config.CATEGORICAL_FEATURES,
    config.TARGET,
    config.TEST_SIZE
  )

  dataset = pd.read_csv(config.PATH_TO_DATASET)

  pipeline.fit(dataset)
  score, f1_score, matrix = pipeline.evaluate_model()
  print(f'Score: {score*100:.2f}%')
  print(f'F1 Score: {f1_score*100:.2f}%')
  print(f'Confusion matrix:\n{matrix}')
