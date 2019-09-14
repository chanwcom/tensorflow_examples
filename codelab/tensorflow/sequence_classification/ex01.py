#!/usr/bin/python3

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



class BinaryClassifier(tf.keras.Model):
    def __init__(self, dim, dropout_rate):
        super(BinaryClassifier, self).__init__()
        self._dense0 = tf.keras.layers.Dense(dim, activation="relu")
        self._dropout = tf.keras.layers.Dropout(0.5)
        self._dense1 = tf.keras.layers.Dense(dim, activation="relu")
        self._softmax = tf.keras.layers.Softmax(-1)

    def call(self, inputs, training=False):
        out = self._dense0(inputs)
        out = self._dropout(out, training=training)
        out = self._dense1(out)
        
        return self._softmax(out) 


binary_classifier = BinaryClassifier(5, 0.5)
y = binary_classifier.predict(np.array([[0.0, 1.0]]))
print (y)

