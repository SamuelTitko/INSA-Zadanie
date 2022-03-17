from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PATH_TO_DATASET = 'dataset.csv'

CLASSIFIER = RandomForestClassifier()

NUMERICAL = [
  'Age',
  'Fare'
]

CATEGORICAL = [
  'PassengerId',
  'Survived',
  'Pclass',
  'Sex',
  'Cabin',
  'Embarked',
  'SibSp',
  'Parch'
]

NUMERICAL_FEATURES = [
  'Age'
]

CATEGORICAL_FEATURES = [
  'Cabin',
  'Pclass',
  'Sex',
]

FEATURES = [
  'Sex',
  'Age',
  'Pclass',
  'Cabin'
]

TARGET = 'Survived'
