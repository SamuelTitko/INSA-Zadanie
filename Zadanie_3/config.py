from sklearn.ensemble import RandomForestClassifier


PATH_TO_DATASET = 'dataset/dataset.csv'

TEST_SIZE = 0.2

NUMERICAL_FEATURES = ['Age']

CATEGORICAL_FEATURES = ['Cabin', 'Pclass','Sex',]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET = ['Survived']

CLASSIFIER = RandomForestClassifier()