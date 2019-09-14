#!/usr/bin/python3

from __future__ import absolute_import

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from data_queue import DataQueue

np.random.seed(0)

# We once considered subclassing the keras.Model rather than using the
# functional API. But we abandoned that idea for two reasons:
#
#  1. Could not find a way to specify the batch size for a stateful
#     LSTM. For a stateful LSTM, the following requirement should be met.
#     1) If using a Sequential model, specify the batch size by passing 
#        a `batch_input_shape` argument to your first layer.
#     2) If using the functional API, specify the batch size by passing 
#        a `batch_shape` argument to your Input layer.
#     However, in the case of subclassing, we could not find a way.
#   
#  2. It is mentioned that the functional API is preferred to subclassing 
#     in most  cases.
#     https://keras.io/models/about-keras-models/
def CreateSequencePredictorModel(batch_size=500, cell_size=10):
    """Create the sequence prediction model.

    Args:

    Returns:
        A created Keras model for sequence prediction.
    """
    inputs= keras.layers.Input(batch_shape=(batch_size, None, 1))
    lstm = keras.layers.LSTM(cell_size, return_sequences=True,
            stateful=True)(inputs)
    dense = keras.layers.Dense(1)(lstm)

    return keras.models.Model(inputs=inputs, outputs=dense)

sequence_predictor = CreateSequencePredictorModel(batch_size=500, cell_size=10)
train_dataset = DataQueue(5000, 500).dataset

sequence_predictor.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(lr=0.05),
        metrics=['mse'])
sequence_predictor.fit(train_dataset, epochs=5)

sequence_predictor_single_batch = CreateSequencePredictorModel(
        batch_size=1, cell_size=10)
sequence_predictor_single_batch.set_weights(sequence_predictor.get_weights())
sequence_predictor_single_batch.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(lr=0.05),
        metrics=['mse'])

print (sequence_predictor_single_batch.predict(
    [[[1.0], [0.0], [-1.0], [-0.0], [1.0], [0.0]]]))
