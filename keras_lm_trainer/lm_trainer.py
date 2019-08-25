#!/usr/bin/python3

# This program requires at least tensorflow 

# pylint: disable=import-error, no-name-in-module, invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import tensorflow as tf
# TODO(chanw.com) Finds a way to assert that the current tf version is >= 2.0.

from speech.trainer.data_queue import word_id_example_queue_config_pb2 
from speech.trainer.data_queue.word_id_example_queue import WordIdExampleQueue
#from speech.trainer.keras_lm_trainer.lm_model import LstmLmModel

class LstmLmModel(tf.keras.Model):
    # TODO TODO(chanw.com)
    # There should be a way for the Example queue to tell the target size.
    def __init__(self, batch_size, target_size, cell_size, dropout_rate):
        super(LstmLmModel, self).__init__()

        self._embedding = tf.keras.layers.Embedding(target_size, cell_size,
            batch_input_shape=(batch_size, None))

        #self._dropout_0 = tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        self._lstm_0 = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, stateful=False)
        #self._dropout_1 = tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        #self._lstm1 = tf.keras.layers.CuDNNLSTM(
        #    cell_size, return_sequences=True, stateful=True),
        #self._dropout_2 =  tf.keras.layers.Dropout(
        #    dropout_rate, noise_shape=(input_shape[0], 1, cell_size)),
        self._softmax = tf.keras.layers.Dense(
            target_size, activation="softmax", )


    def call(self, inputs, training=False):
        output = self._embedding(inputs)
        output = self._lstm_0(output)
        output = self._softmax(output)

        return (output)

config = word_id_example_queue_config_pb2.WordIdExampleQueueConfig()

# TODO TODO (The following needs to be fixed)
config.file_name = "/home/chanwcom/chanwcom_local_repository/star_organization/speech01/speech/speech/trainer/keras_lm_trainer/data/ptb.train.txt"
config.batch_size = 20
config.randomize_order = False

word_id_example_queue = WordIdExampleQueue(config)

#TODO TODO(chanw.com) Adds a routine to construct the example queue.

#target_size = 10002
target_size = 10010
cell_size = 256
dropout_rate = 0.5

lm_model = LstmLmModel(config.batch_size, target_size, cell_size, dropout_rate)

lm_model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy'])
# Attach the tf.data dataset.

#for data in word_id_example_queue.dataset:
#    print (data[0].numpy().shape)

#print (lm_model.summary())
lm_model.fit(word_id_example_queue.dataset, epochs=5)
    #callbacks=[MyCustomCallback(lm_model)])


#loss = lm_model.evaluate(test_dataset)
#print ("perplexity is {0}.".format(np.exp(loss)))


if 0:
    fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1,
        callbacks=None, validation_data=None, validation_steps=None,
        validation_freq=1, class_weight=None, max_queue_size=10, workers=1,
        use_multiprocessing=False, shuffle=True, initial_epoch=0)

    num_steps = 30
    num_epochs = 10
    batch_size = 10

    steps_per_epoch = train_data_size // batch_size - 1
    model.fit_generator(train_data_generator, 
                        (train_data)//(batch_size*num_steps),   # steps per epoch
                        num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps),
                        callbacks=[checkpointer])
