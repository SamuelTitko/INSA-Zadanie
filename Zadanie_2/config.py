from sklearn.ensemble import RandomForestClassifier


PATH_TO_DATASET = 'dataset.csv'

CLASSIFIER = RandomForestClassifier()

NUMERICAL = [
  'Age',
  'Fare'
]

CATEGORICAL = [
  'Survived',
  'Pclass',
  'Sex',
  'Cabin',
  'Embarked',
  'SibSp',
  'Parch'
]

FEATURES = [
  'Age',
  'Pclass',
  'Cabin',
  'Sex',
]

NUMERICAL_FEATURES = [
  'Age'
]

CATEGORICAL_FEATURES = [
  'Pclass',
  'Cabin',
  'Sex',
]

TARGET = ['Survived']
