if __name__ == '__main__':
  import config
  import pandas as pd
  from preprocessing import Pipeline

  pipeline = Pipeline(
    config.CLASSIFIER,
    config.NUMERICAL,
    config.CATEGORICAL,
    config.NUMERICAL_FEATURES,
    config.CATEGORICAL_FEATURES,
    config.FEATURES,
    config.TARGET,
  )

  dataset = pd.read_csv('dataset/dataset.csv')

  pipeline.fit(dataset)
  score, matrix = pipeline.evaluate_model()
  print(f'Accuracy: {score*100:.2f}%')
  print(f'Confusion matrix:\n{matrix}')