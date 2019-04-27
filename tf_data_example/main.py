#!/usr/bin/python
'''An example of estimating the power coeff using tf.data.'''

# pylint: disable=unexpected-keyword-arg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

__author__ = 'Chanwoo Kim(chanw.com@samsung.com)'


def get_batch():
    '''Gets the next batch from the training set.

    The training set returns a tuple of (input, target_value).
    The target value is input ** 1.0 / 15.0 in this sample example.

    Params:
        None.

    Returns:
        A tuple of the next example.
    '''
    data_size = 1000
    true_power_coeff = 1.0 / 15.0

    def data_generation(data_size, true_power_coeff):
        '''A generator for generating a tuple of (input, target_value).'''
        for data in np.random.uniform(0.0, 100.0, data_size):
            yield (data, data ** true_power_coeff)

    dataset = tf.data.Dataset.from_generator(
        data_generation, (tf.float32, tf.float32),
        args=[data_size, true_power_coeff])
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def inference(input_batch):
    '''Creates the inference graph.

    Params:
        input_batch: The input batch.

    Returns:
        A tensor containing the output batch.
    '''
    with tf.variable_scope('inference'):
        alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=0.0)
        output_batch = input_batch ** alpha

    return output_batch


def main():
    '''The main entry function.'''
    next_element = get_batch()
    inference_output = inference(next_element[0])
    mse_loss = tf.losses.mean_squared_error(next_element[1], inference_output)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(mse_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(train_step)

            if i % 100 == 0:
                with tf.variable_scope('inference', reuse=True):
                    alpha = tf.get_variable('alpha', dtype=tf.float32)
                    print ('global step: {0}, the estimated alpha: {1}'.format(
                        i, sess.run(alpha)))

if __name__ == '__main__':
    main()
