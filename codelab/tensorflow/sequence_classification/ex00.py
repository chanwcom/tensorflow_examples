#!/usr/bin/python

import tensorflow as tf

from tensorflow.keras.datasets import imdb

top_words=5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print (X_train)
print ("--------------")
print (X_train.shape)
print ("--------------")

print (X_test.shape)

print (y_train)
