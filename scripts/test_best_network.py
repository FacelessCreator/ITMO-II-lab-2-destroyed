from train_network import train_network

from numpy import NaN
from tensorflow.keras import optimizers, losses
import pandas as pd
from sklearn.model_selection import train_test_split

NETWORK_ACCURACIES_FILEPATH = 'build/network_accuracies.csv'

METRICS_NAME = 'mae'

networks_data = pd.read_csv(NETWORK_ACCURACIES_FILEPATH)

minimalize_column_name = METRICS_NAME + '_score'
minimal_value = networks_data[minimalize_column_name].min()
minimal_row = networks_data[networks_data[minimalize_column_name] == minimal_value].iloc[0]

print('Best {} metrics is {}'.format(METRICS_NAME, minimal_value))
print(minimal_row)

"""
    DATA PREPARAION
"""

FILTERED_TRAIN_FILEPATH = 'build/filtered_train.csv'
FILTERED_TEST_FILEPATH = 'build/filtered_test.csv'
Y_LABELS = ['SalePrice']
TEST_SIZE = 0.2

train_data = pd.read_csv(FILTERED_TRAIN_FILEPATH)

X = train_data.drop(Y_LABELS, axis=1)
Y = train_data.drop(train_data.columns.difference(Y_LABELS), axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = TEST_SIZE)

test_data = pd.read_csv(FILTERED_TEST_FILEPATH)

"""
    TRAINING
"""

RANDOM_SEED = 40

model = train_network(X_train, Y_train, RANDOM_SEED, minimal_row['start_layer_size'], minimal_row['hidden_layer_size'], minimal_row['epochs_count'], minimal_row['batch_size'], minimal_row['optimizer'], minimal_row['loss'], minimal_row['activation'])

"""
    PREDICTING
"""

BEST_ANSWERS_FILEPATH = 'build/best_answers.csv'

preds = model.predict(test_data)
output = pd.DataFrame(data=preds, index=test_data['Id'], columns=Y_LABELS)
output.to_csv(BEST_ANSWERS_FILEPATH)
