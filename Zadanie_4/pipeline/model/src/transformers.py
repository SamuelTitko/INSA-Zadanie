import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class CategoricalTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, variables=None):
    self.variables = variables
    self.encoder = OneHotEncoder(sparse=False)

  def fillna(self, dataset):
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x if pd.isna(x) else str(x)[0])
    for variable in self.variables:
      dataset[variable] = dataset[variable].ffill()
      most_common = dataset[variable].value_counts()
      dataset[variable] = dataset[variable].fillna(most_common.index[0])
    return dataset

  def fit(self, X, y=None):
    dataset = self.fillna(X.copy())
    self.encoder.fit(dataset[self.variables])
    return self

  def transform(self, X):
    dataset = self.fillna(X.copy())
    dataset[self.encoder.get_feature_names_out()] = self.encoder.transform(dataset[self.variables])
    dataset.drop(self.variables, axis=1, inplace=True)
    return dataset


class NumericalTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, variables=None):
    self.variables = variables
    self.scaler = MinMaxScaler()

  def fillna(self, dataset):
    for variable in self.variables:
      dataset[variable] = dataset[variable].fillna(dataset[variable].mean())
    return dataset

  def fit(self, X, y=None):
    dataset = self.fillna(X.copy())
    self.scaler.fit(dataset[self.variables])
    return self

  def transform(self, X):
    dataset = self.fillna(X.copy())
    dataset[self.variables] = self.scaler.transform(dataset[self.variables])
    return dataset
