import sys

import pandas as pd

TRAIN_FILEPATH = 'src/train.csv'
TEST_FILEPATH = 'src/test.csv'

FILTERED_TRAIN_FILEPATH = 'build/filtered_train.csv'
FILTERED_TEST_FILEPATH = 'build/filtered_test.csv'

train_data = pd.read_csv(TRAIN_FILEPATH)
test_data = pd.read_csv(TEST_FILEPATH)

def check_missing_values(data):
    list = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            sum = data[feature].isna().sum()
            type = data[feature].dtype
            list.append(feature)
    if (len(list) > 0):
        sys.exit('Cannot clean missing values:\n{}'.format(list))

def delete_rare_params(data):
    dropped_columns = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            sum = data[feature].isna().sum()
            if (sum >= content.size/2):
                dropped_columns.append(feature)
                data.drop(feature, axis=1, inplace=True)
    return dropped_columns

def nan_filler(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.mean()) # here we can switch between some normal values: mean, median, mode
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes+1

dropped_columns = delete_rare_params(train_data)
test_data.drop(dropped_columns, axis=1, inplace=True)

nan_filler(train_data)
nan_filler(test_data)

check_missing_values(train_data)
check_missing_values(test_data)

train_data.to_csv(FILTERED_TRAIN_FILEPATH)
test_data.to_csv(FILTERED_TEST_FILEPATH)
