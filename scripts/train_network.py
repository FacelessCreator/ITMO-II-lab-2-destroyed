import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

def train_network(X_train, Y_train, random_seed, start_layer_size=10, hidden_layer_size=0, epochs_count=10, batch_size=10, optimizer="adam", loss="mse", activation=activations.relu, metrics=['mae']):
    answer_columns_count = Y_train.shape[1]
    if (hidden_layer_size <= 0):
        model = keras.Sequential(
            [
                layers.Dense(start_layer_size, activation=activation),
                layers.Dense(answer_columns_count, activation=activation),
            ]
        )
    else:
        model = keras.Sequential(
            [
                layers.Dense(start_layer_size, activation=activation),
                layers.Dense(hidden_layer_size, activation=activation),
                layers.Dense(answer_columns_count, activation=activation),
            ]
        )
    tf.random.set_seed(random_seed)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs_count, verbose=0)
    return model