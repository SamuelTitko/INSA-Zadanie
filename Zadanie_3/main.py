import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config
from transformers import NumericalTransformer, CategoricalTransformer


pipeline = Pipeline([
  ('categorical transformer', CategoricalTransformer(config.CATEGORICAL_FEATURES)),
  ('numerical transformer', NumericalTransformer(config.NUMERICAL_FEATURES)),
  ("classifier model", config.CLASSIFIER),
])

dataset = pd.read_csv(config.PATH_TO_DATASET)
x_train, x_test, y_train, y_test = train_test_split(
  dataset[config.FEATURES],
  dataset[config.TARGET],
  test_size=config.TEST_SIZE
)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print(f'Score: {pipeline.score(x_test, y_test)*100:.2f}%')
print(f'F1 Score: {f1_score(y_test, y_pred)*100:.2f}%')
print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')