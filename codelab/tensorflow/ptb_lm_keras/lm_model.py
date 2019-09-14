#!/usr/bin/python3

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(0)

def CreateLMModel(batch_size, cell_size, target_size, dropout_rate): 
    inputs= keras.layers.Input(batch_shape=(batch_size, None, 1))
    embedding = keras.layers.Embedding(
        target_size, size)(inputs)
    dropout0 = keras.layers.Dropout(
        dropout_rate, noise_shape=(input_shape[0], 1, size))(embedding)
    lstm0 =  keras.layers.LSTM(
        size, return_sequences=True, stateful=True, batch_size=batch_size)(
            dropout0)
    dropout1 = keras.layers.Dropout(
        dropout_rate, noise_shape=(input_shape[0], 1, size))(lstm0)
    lstm1 = keras.layers.LSTM((
        size, return_sequences=True, stateful=True)(dropout1)
    dropout2 = keras.layers.Dropout(
        dropout_rate, noise_shape=(input_shape[0], 1, size))(sltm1)
    softmax = keras.layers.Dense(
        target_size, activation="softmax")(dropout2)

    return keras.models.Model(inputs=inputs, outputs=softmax)


lm_model = LMModel(256, 10002)
lm_model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])
# Attach the tf.data dataset.

lm_model.fit(train_dataset, epochs=5,
        callbacks=[MyCustomCallback(lm_model)])
print (lm_model.summary())

loss = lm_model.evaluate(test_dataset)
print ("perplexity is {0}.".format(np.exp(loss)))
