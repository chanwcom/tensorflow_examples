#!/usr/bin/python3 
__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# LSTM for sequence classification in the IMDB dataset
import numpy

import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    path="imdb.npz", num_words=top_words)
# Truncates and pads input sequences.
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


class SequenceClassifier(tf.keras.Model):
  def __init__(self, num_words, vector_size, batch_size):
    super(SequenceClassifier, self).__init__()

    self._embedding = Embedding(
        num_words, vector_size, batch_input_shape=(batch_size, None))
    self._lstm = tf.keras.layers.LSTM(10, return_state=True)
    self._dense = Dense(1, activation='sigmoid')

  def call(self, inputs, training=False):
    out = self._embedding(inputs)
    out = self._lstm(out)
    return self._dense(out[0])


# create the model
embedding_vecor_length = 16 
model = SequenceClassifier(top_words, embedding_vecor_length, batch_size=50)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=50)

print(model.summary())

# Final evaluation of the model
scores = model.evaluate(X_test[:100,], y_test[:100,], verbose=0, batch_size=50)

print("Accuracy: %.2f%%" % (scores[1]*100))

model.save_weights("classifier.h5")
