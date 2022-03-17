import pandas as pd

dataset_train = pd.read_csv('dataset/train.csv')
dataset_test_x = pd.read_csv('dataset/test.csv')
dataset_test_y = pd.read_csv('dataset/gender_submission.csv')
dataset_test_x[dataset_test_y.keys()] = dataset_test_y.values
dataset = dataset_train.append(dataset_test_x)

dataset.to_csv('dataset/dataset.csv', index=False)