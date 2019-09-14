#!/usr/bin/python3

# Required version: Tensorflow 2.0.0-beta1
# TODO(chanw.com) Implement a version checking routine.

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)


def generate_data(example_size):
    """Generates two-dimensional training data from two classes. 
    
    The number of examples in the data is "example_size". Each example is
    randomly generated from one of two classes. Data belonging to each
    class are generated using two two-dimensional Gaussian distribution
    with different means and covariance matrices.

    Args:
        example_size: The number of examples in the generated data.

    Returns:
        A tuple of (x, class_id). "x" contains the two-dimensional feature data
        and class_id contains the label id, which is either zero or one.
    """


    # A placeholder for data.
    x = np.zeros((example_size, 2))

    # Randomly selects one of two classes for each example.
    class_id = np.random.randint(2, size=example_size)

    # Generates data for the class 0.
    class_0_idx = np.where(class_id == 0)
    mean0 = [0.0, 1.0]
    cov0 = [[0.4, 0.0], [0.0, 1.0]]
    x[class_0_idx[0], :] = np.random.multivariate_normal(
            mean0, cov0,  class_0_idx[0].shape[0])

    # Generates data for the class 0.
    class_1_idx = np.where(class_id == 1)
    mean1 = [1.0, 2.0]
    cov1 = [[1.0, 0.0], [0.0, 0.4]]
    x[class_1_idx[0], :] = np.random.multivariate_normal(
            mean1, cov1, class_1_idx[0].shape[0])

    return (x, class_id)

train_data = generate_data(10000)
test_data = generate_data(100)

train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.batch(100)

test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
test_dataset = test_dataset.batch(10)


class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(MyCustomCallback, self).__init__()
        self._model = model

    def on_epoch_begin(self, epoch, logs=None):
        print ("{0}-th epoch started".format(epoch))


class BinaryClassifier(tf.keras.Model):
    def __init__(self, dim, dropout_rate=0.5):
        super(BinaryClassifier, self).__init__()
        self._dense0 = tf.keras.layers.Dense(dim, activation="relu")
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._softmax = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        out = self._dense0(inputs)
        out = self._dropout(out) 
        return self._softmax(out) 


binary_classifier = BinaryClassifier(10)

# Compiles the model.
#
# Note that 'sparse_categorical_crossentropy' is used for multi-class 
# classification.
binary_classifier.compile(
        loss='binary_crossentropy', 
        optimizer='adam',
        metrics=['accuracy'])
binary_classifier.fit(train_dataset, epochs=5,
        callbacks=[MyCustomCallback(binary_classifier)])

print (binary_classifier.summary())

binary_classifier.evaluate(test_dataset)
output = binary_classifier.predict(test_dataset)
decision = np.argmax(output, axis=1)

def scatter_plot_dataset(data, ground_truth, decision):
    index = np.where(np.logical_and(ground_truth ==0, decision ==0))
    plt.scatter(data[index][:, 0], data[index][:, 1], alpha=0.8, c="r",
            label='Label:0, Decision: 0' )
    index = np.where(np.logical_and(ground_truth ==0, decision ==1))
    plt.scatter(data[index][:, 0], data[index][:, 1], alpha=0.8, c="c", 
            label='Label:0, Decision: 1' )
    index = np.where(np.logical_and(ground_truth ==1, decision ==0))
    plt.scatter(data[index][:, 0], data[index][:, 1], alpha=0.8, c="y",
            label='Label:1, Decision: 0' )
    index = np.where(np.logical_and(ground_truth ==1, decision ==1))
    plt.scatter(data[index][:, 0], data[index][:, 1], alpha=0.8, c="g",
            label='Label:1, Decision: 1' )
    plt.grid()
    plt.legend()
    plt.show()

scatter_plot_dataset(test_data[0], test_data[1], decision)
