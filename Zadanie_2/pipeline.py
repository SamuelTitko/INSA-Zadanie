import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Pipeline:
  def __init__(self, model, numerical_features, categorical_features, target, test_size=0.30):
    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None

    self.scaler = MinMaxScaler()
    self.encoder = OneHotEncoder(sparse=False)
    self.model = model

    self.numerical_features = numerical_features
    self.categorical_features = categorical_features
    self.target = target
    self.required_columns = numerical_features + categorical_features + target

    self.test_size = test_size

  def fillna_numerical(self, dataset):
    for numerical in self.numerical_features:
      dataset[numerical] = dataset[numerical].fillna(dataset[numerical].mean())

  def fillna_categorical(self, dataset):
    for categorical in self.categorical_features:
      dataset[categorical] = dataset[categorical].ffill()
      most_common = dataset[categorical].value_counts()
      dataset[categorical] = dataset[categorical].fillna(most_common.index[0])

  def fit(self, X):
    dataset = X[self.required_columns]
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x if pd.isna(x) else str(x)[0])

    self.fillna_numerical(dataset)
    self.fillna_categorical(dataset)

    dataset[self.numerical_features] = self.scaler.fit_transform(dataset[self.numerical_features])
    dataset[self.encoder.get_feature_names_out()] = self.encoder.fit_transform(dataset[self.categorical_features])

    dataset.drop(self.categorical_features, axis=1, inplace=True)

    x, y = dataset.drop(self.target, axis=1), dataset[self.target]
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size)
    self.model.fit(self.x_train, self.y_train)

  def transform(self, X):
    dataset = X[self.required_columns]
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x if pd.isna(x) else str(x)[0])

    self.fillna_numerical(dataset)
    self.fillna_categorical(dataset)

    dataset[self.numerical_features] = self.scaler.transform(dataset[self.numerical_features])
    dataset[self.encoder.get_feature_names_out()] = self.encoder.transform(dataset[self.categorical_features])

    dataset.drop(self.categorical_features, axis=1, inplace=True)

    return dataset

  def predict(self, X):
    return self.model.predict(self.transform(X))

  def evaluate_model(self):
    y_pred = self.model.predict(self.x_test)
    return [
      self.model.score(self.x_test, self.y_test),
      f1_score(self.y_test, y_pred),
      confusion_matrix(self.y_test, y_pred)
    ]
