from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import tensorflow as tf

class LstmLmModel(tf.keras.Model):
    # TODO TODO(chanw.com)
    # There should be a way for the Example queue to tell the target size.
    def __init__(self, batch_size, target_size, cell_size, dropout_rate):
        super(LstmLmModel, self).__init__()

        self._embedding = tf.keras.layers.Embedding(
            target_size, cell_size, batch_input_shape=(batch_size, None)),
        #self._dropout_0 = tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        self._lstm_0 = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, stateful=True,
            batch_size=batch_size),
        #self._dropout_1 = tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        #self._lstm1 = tf.keras.layers.CuDNNLSTM(
        #    cell_size, return_sequences=True, stateful=True),
        #self._dropout_2 =  tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        self._softmax = tf.keras.layers.Dense(
            target_size, activation="softmax")


    def call(self, inputs, training=False):

        output = self._embedding(inputs)
        output = self._lstm_0(output)
        output = self._softmax(output)

        return (output)
