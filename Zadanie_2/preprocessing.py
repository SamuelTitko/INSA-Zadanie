import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Pipeline:
  def __init__(self, model, numerical, categorical, numerical_features, categorical_features, features, target, test_size=0.30):
    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None

    self.scalers = dict()
    self.encoders = dict()
    self.model = model

    self.numerical = numerical
    self.categorical = categorical
    self.numerical_features = numerical_features
    self.categorical_features = categorical_features
    self.features = features
    self.target = target
    self.required_columns = self.features + [self.target]

    self.test_size = test_size

  def fillna_numerical(self, dataset):
    for numerical in self.numerical_features:
      dataset[numerical] = dataset[numerical].fillna(dataset[numerical].mean())

  def scale_numerical(self, dataset):
    for numerical in self.numerical_features:
      scaler = MinMaxScaler()
      dataset[[numerical]] = scaler.fit_transform(dataset[[numerical]])
      self.scalers[numerical] = scaler

  def fillna_categorical(self, dataset):
    for categorical in self.categorical_features:
      dataset[categorical] = dataset[categorical].ffill()
      most_common = dataset[categorical].value_counts()
      dataset[categorical] = dataset[categorical].fillna(most_common.index[0])

  def encode_categorical(self, dataset):
    for categorical in self.categorical_features:
      encoder = OneHotEncoder()
      dataset[encoder.get_feature_names_out()] = encoder.fit_transform(dataset[categorical].to_numpy().reshape(-1, 1)).toarray()
      dataset.drop([categorical], axis=1, inplace=True)
      self.encoders[categorical] = encoder

  def fit(self, dataset):
    dataset = dataset[self.required_columns].copy()
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x if pd.isna(x) else str(x)[0])

    self.fillna_numerical(dataset)
    self.scale_numerical(dataset)
    self.fillna_categorical(dataset)
    self.encode_categorical(dataset)

    x, y = dataset.drop(self.target, axis=1), dataset[self.target]
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size)
    self.model.fit(self.x_train, self.y_train)

  def transform(self, dataset):
    dataset = dataset[self.required_columns].copy()
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x if pd.isna(x) else str(x)[0])

    self.fillna_numerical(dataset)
    self.fillna_categorical(dataset)

    for numerical in self.numerical_features:
      dataset[[numerical]] = self.scalers[numerical].transform(dataset[[numerical]])
    for categorical in self.categorical_features:
      column_names = self.encoders[categorical].get_feature_names_out()
      values = dataset[categorical].to_numpy().reshape(-1, 1)
      dataset[column_names] = self.encoders[categorical].fit_transform(values).toarray()
      dataset.drop([categorical], axis=1, inplace=True)
    return dataset

  def predict(self, dataset):
    return self.model.predict(self.transform(dataset))

  def evaluate_model(self):
    return self.model.score(self.x_test, self.y_test), confusion_matrix(self.y_test, self.model.predict(self.x_test))
    
