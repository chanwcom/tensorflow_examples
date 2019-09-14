#!/usr/bin/python3

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import numpy as np
import tensorflow as tf

def _create_cosine_sequence(period_samples, length_samples):
    assert period_samples > 0.0
    assert length_samples > 0.0
    x = np.arange(length_samples)
    y = np.cos(2.0 * np.pi / period_samples * x)

    return np.expand_dims(y, axis=1)


def _cosine_sequence_generator(num_examples):
    assert num_examples > 0

    periods = np.random.uniform(3.9, 4.1, num_examples)
    lengths = np.random.uniform(20, 40, num_examples)

    for i in range(num_examples):
        yield _create_cosine_sequence(periods[i], lengths[i])


class DataQueue(object):
    def __init__(self, num_examples, batch_size):
        dataset = tf.data.Dataset.from_generator(
            lambda: _cosine_sequence_generator(num_examples=num_examples), 
            (tf.float64), (tf.TensorShape([None, 1])))

        # Performs padded batching.
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=tf.TensorShape([None, 1]),
            drop_remainder=True)

        # Converts batch into (inputs, targets) format.
        #
        # The target contains future values created by shifting the sequence
        # by one-sample.
        dataset = dataset.map(lambda batch: (batch[:, :-1, :], batch[:, 1:, :]))

        self._dataset = dataset


    @property
    def dataset(self):
        return self._dataset
