#!/usr/bin/python3
'''The main function to train an RNN LM from the PTB corpus.
This program is originally downloaded from the following git hub.
https://github.com/MilkKnight/RNN_Language_Model
Modified and improved by Chanwoo Kim.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

from data_prepare import gen_vocab
from data_prepare import gen_id_seqs
from rnn_lm import RNNLM

import tensorflow as tf
# It seems that there are some little bugs in tensorflow 1.4.1.
# You can find more details in
# https://github.com/tensorflow/tensorflow/issues/12414
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO(chanw.com)
# 1. Use the tf.train.MonitoredTrainingSession.
# 2. Use tf.summary for tensorboard display.
# 3. Use the sharded TFRecords.

# Set TRAIN to true will build a new model
TRAIN = True

def main(args):
    start_time = time.time()

    # To indicate your test corpus
    test_file = "./gap_filling_exercise/gap_filling_exercise"

    if not os.path.isfile("data/vocab"):
        gen_vocab("ptb/train")
    if not os.path.isfile("data/train.ids"):
        gen_id_seqs("ptb/train")
        gen_id_seqs("ptb/valid")

    with open("data/train.ids") as fp:
        num_train_samples = len(fp.readlines())
    with open("data/valid.ids") as fp:
        num_valid_samples = len(fp.readlines())

    with open("data/vocab") as vocab:
        vocab_size = len(vocab.readlines())

    def create_model(sess):
        model = RNNLM(
            vocab_size=vocab_size, batch_size=64, num_epochs=80,
            check_point_step=100, num_train_samples=num_train_samples,
            num_valid_samples=num_valid_samples, num_layers=2,
            num_hidden_units=600, initial_learning_rate=1.0,
            final_learning_rate=0.0005, max_gradient_norm=5.0)
        sess.run(tf.global_variables_initializer())
        return model

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction)

    if TRAIN:
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = create_model(sess)
            saver = tf.train.Saver()
            model.batch_train(sess, saver)

    tf.reset_default_graph()

    with tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = create_model(sess)
        saver = tf.train.Saver()
        saver.restore(sess, "model/best_model.ckpt")
        predict_id_file = os.path.join("data", test_file.split("/")[-1]+".ids")
        if not os.path.isfile(predict_id_file):
            gen_id_seqs(test_file)
        model.predict(sess, predict_id_file, test_file, verbose=args.verbose)

    print ("========================================")
    print ("Time elapsed is {0}".format(time.time() - start_time))
    print ("========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_memory_fraction',
        default=0.9,
        help=('The hard upper bound on the amount of GPU memory that will '
              'be used by the process on each GPU on the same machine.'),
        type=float)
    parser.add_argument(
        '--verbose',
        default=True,
        help='If enabled, the prediction result is shown.',
        type=bool)

    args = parser.parse_args()
    main(args)

