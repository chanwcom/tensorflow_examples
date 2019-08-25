"""A module for unit testing word_id_example_queue."""

# pylint: disable=import-error, no-name-in-module, invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import tempfile
import unittest
import numpy as np
import tensorflow as tf

from speech.trainer.data_queue import word_id_example_queue_config_pb2
from speech.trainer.data_queue.word_id_example_queue import WordIdExampleQueue


class ExampleQueueTest(unittest.TestCase):
    """A class for testing classes and methods in word_id_example_queue.py."""

    def __init__(self, *args, **kwargs):
        """Creates the input text file.

        The following is the contents of the input text file.
          * my name is chanwoo kim
          * another name is chanwcom kim
          * everyone has a name
          * everyone has a car
          * my car has a name
        """
        super(ExampleQueueTest, self).__init__(*args, **kwargs)

        # Creates a temporary file and stores the contents of the input
        # text file.
        self._input_text_file = tempfile.NamedTemporaryFile(mode="w+b")
        # The following three sentences are used as the input file.
        with open(self._input_text_file.name, "wt") as f:
            f.write("my name is chanwoo kim\n")
            f.write("another name is chanwcom kim\n")
            f.write("everyone has a name\n")
            f.write("everyone has a car\n")
            f.write("my car has a name\n")

    def test_batch_size_one_test(self):
        """Tests the case when the batch size is one.

        The input text is specified in the above __init__ method.
        """

        config = word_id_example_queue_config_pb2.WordIdExampleQueueConfig()
        config.file_name = self._input_text_file.name
        config.batch_size = 1
        config.randomize_order = False

        word_id_example_queue = WordIdExampleQueue(config)

        expected_output_list = []
        expected_output_list.append(np.array([2, 10, 3, 8, 13, 9, 1]))
        expected_output_list.append(np.array([2, 11, 3, 8, 12, 9, 1]))
        expected_output_list.append(np.array([2, 7, 5, 4, 3, 1]))
        expected_output_list.append(np.array([2, 7, 5, 4, 6, 1]))
        expected_output_list.append(np.array([2, 10, 6, 5, 4, 3, 1]))

        actual_output_list = []
        with tf.Session() as sess:
            sess.run(word_id_example_queue.initializer)
            while True:
                try:
                    actual_output_list.append(
                        sess.run(word_id_example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # The output contains five batches.
        self.assertEqual(len(expected_output_list), len(actual_output_list))

        # Each batch is expected to contain one example, because the batch
        # size is one.
        self.assertEqual(1, actual_output_list[0].shape[0])

        for index, expected_output in enumerate(expected_output_list):
            self.assertTrue(
                (expected_output == actual_output_list[index]).all())

    def test_batch_size_two_test(self):
        """Tests the case when the batch size is two.

        The input text is specified in the above __init__ method.
        """

        config = word_id_example_queue_config_pb2.WordIdExampleQueueConfig()
        config.file_name = self._input_text_file.name
        config.batch_size = 2
        config.randomize_order = False

        word_id_example_queue = WordIdExampleQueue(config)

        expected_output_list = []
        expected_output_list.append(np.array([
            [2, 10, 3, 8, 13, 9, 1],
            [2, 11, 3, 8, 12, 9, 1],
        ]))
        expected_output_list.append(np.array([
            [2, 7, 5, 4, 3, 1],
            [2, 7, 5, 4, 6, 1],
        ]))

        actual_output_list = []
        with tf.Session() as sess:
            sess.run(word_id_example_queue.initializer)
            while True:
                try:
                    actual_output_list.append(
                        sess.run(word_id_example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # The output contains two batches. 5 // 2 = 2.
        self.assertEqual(2, len(actual_output_list))

        # Each batch is expected to contain two examples, because the batch
        # size is two.
        self.assertEqual(2, actual_output_list[0].shape[0])

        # Compares the contents of the actual outputs with the expected output.
        self.assertTrue((expected_output_list[0] ==
                         actual_output_list[0]).all())
        self.assertTrue((expected_output_list[1] ==
                         actual_output_list[1]).all())


if __name__ == '__main__':
    unittest.main()
