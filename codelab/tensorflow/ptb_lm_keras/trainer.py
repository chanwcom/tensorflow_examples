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

config = word_id_example_queue_config_pb2.WordIdExampleQueueConfig()
config.file_name = self._input_text_file.name
config.batch_size = 1
config.randomize_order = False

word_id_example_queue = WordIdExampleQueue(config)

lm_model = LMModel(256, 10002)
lm_model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])

lm_model.fit(word_id_example_queue.dataset, epochs=5,
        callbacks=[MyCustomCallback(lm_model)])
print (lm_model.summary())

loss = lm_model.evaluate(test_dataset)
print ("perplexity is {0}.".format(np.exp(loss)))
