import time

from tensorflow.python.keras.activations import linear

from train_network import train_network

from tensorflow.keras import optimizers, losses, activations
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    DATA PREPARAION
"""

FILTERED_TRAIN_FILEPATH = 'build/filtered_train.csv'
Y_LABELS = ['SalePrice']
TEST_SIZE = 0.2

train_data = pd.read_csv(FILTERED_TRAIN_FILEPATH)

X = train_data.drop(Y_LABELS, axis=1)
Y = train_data.drop(train_data.columns.difference(Y_LABELS), axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = TEST_SIZE)

"""
    NETWORKS TESTING
"""

NO_TIME_TO_LEARN_MODE = False # Use this mode to ignore settings brute force

RANDOM_SEED = 40

METRICS = ['mae']

LEARNING_RATE = 0.001

DEFAULT_START_LAYER_SIZE = 400
DEFAULT_HIDDEN_LAYER_SIZE = 400
DEFAULT_EPOCHS_COUNT = 500
DEFAULT_BATCH_SIZE = 50
DEFAULT_OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE)
DEFAULT_LOSS = losses.MeanAbsoluteError()
DEFAULT_ACTIVATION = activations.linear

START_LAYER_SIZES = range(10,500,10)
HIDDEN_LAYER_SIZES = range(10,500,10)
EPOCHS_COUNTS = range(50,1000,50)
BATCH_SIZES = range(10,200,10)

OPTIMIZERS = [optimizers.Adam(learning_rate=LEARNING_RATE), optimizers.SGD(learning_rate=LEARNING_RATE), optimizers.Adadelta(learning_rate=LEARNING_RATE), optimizers.Adamax(learning_rate=LEARNING_RATE), optimizers.RMSprop(learning_rate=LEARNING_RATE)]
LOSSES = [losses.MeanAbsoluteError(), losses.MeanSquaredError(), losses.MeanAbsolutePercentageError(), losses.MeanSquaredLogarithmicError()]
ACTIVATIONS = [activations.relu, activations.exponential, activations.sigmoid, activations.linear, activations.gelu]

network_accuracies = []

model_number = 1
start_time = time.time()

MODELS_COUNT = len(START_LAYER_SIZES)+len(HIDDEN_LAYER_SIZES)+len(EPOCHS_COUNTS)+len(BATCH_SIZES)+len(OPTIMIZERS)+len(LOSSES)+len(ACTIVATIONS)

def train_network_and_remember_scores(experiment_sign, start_layer_size, hidden_layer_size, epochs_count, batch_size, optimizer, loss, activation):
    global X_train, Y_train, X_val, Y_val, network_accuracies, RANDOM_SEED, METRICS, start_time, model_number
    time_delta = time.time() - start_time
    print('Model {}/{} - {:.3f} s since start'.format(model_number, MODELS_COUNT, time_delta))
    model = train_network(X_train, Y_train, RANDOM_SEED, start_layer_size, hidden_layer_size, epochs_count, batch_size, optimizer, loss, activation, METRICS)
    scores = model.evaluate(X_val, Y_val, verbose=0)
    model_params = [start_layer_size, hidden_layer_size, epochs_count, batch_size, type(optimizer).__name__, type(loss).__name__, activation.__name__]
    print('{}: start_layer_size={}, hidden_layer_size={}, epoch_count={}, batch_size={}, optimizer={}, loss={}, activation={}'.format(experiment_sign,*model_params))
    network_accuracies.append([experiment_sign]+model_params+scores)
    model_number += 1

if not NO_TIME_TO_LEARN_MODE:
    for start_layer_size in START_LAYER_SIZES:
        train_network_and_remember_scores('start_layer_size', start_layer_size, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_ACTIVATION)
    for hidden_layer_size in HIDDEN_LAYER_SIZES:
        train_network_and_remember_scores('hidden_layer_size', DEFAULT_START_LAYER_SIZE, hidden_layer_size, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_ACTIVATION)
    for epochs_count in EPOCHS_COUNTS:
        train_network_and_remember_scores('epochs_count', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, epochs_count, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_ACTIVATION)
    for batch_size in BATCH_SIZES:
        train_network_and_remember_scores('batch_size', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, batch_size, DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_ACTIVATION)
    for optimizer in OPTIMIZERS:
        train_network_and_remember_scores('optimizer', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, optimizer, DEFAULT_LOSS, DEFAULT_ACTIVATION)
    for loss in LOSSES:
        train_network_and_remember_scores('loss', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, loss, DEFAULT_ACTIVATION)
    for activation in ACTIVATIONS:
        train_network_and_remember_scores('activation', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, DEFAULT_LOSS, activation)
else:
    train_network_and_remember_scores('no time to learn', DEFAULT_START_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_EPOCHS_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_OPTIMIZER, DEFAULT_LOSS, DEFAULT_ACTIVATION)

time_delta = time.time() - start_time
print('Process completed in {:.3f} s'.format(time_delta))

"""
    SAVING RESULTS
"""

NETWORK_ACCURACIES_FILEPATH = 'build/network_accuracies.csv'
NA_DF_COLUMNS = ['experiment_sign','start_layer_size','hidden_layer_size','epochs_count','batch_size','optimizer','loss','activation', 'loss_score'] + list(map(lambda s: s+'_score', METRICS))

network_accuracies_df = pd.DataFrame(data=network_accuracies, columns=NA_DF_COLUMNS)

network_accuracies_df.to_csv(NETWORK_ACCURACIES_FILEPATH)
