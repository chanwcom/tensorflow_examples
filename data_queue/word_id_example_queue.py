"""A module implementing WordIdExampleQueue.

Note that this version has been tested with tensorflow version 1.14.

This ExampleQueue supports randomization, shuffling, padded batch, etc.
The usage is as follows:

    config = word_id_example_queue_config_pb2.WordIdExampleQueueConfig()
    config.file_name = "input_text_file"
    config.batch_size = 20

    word_id_example_queue = WordIdExampleQueue(config)

    with tf.Session() as sess:
        sess.run(word_id_example_queue.initializer)
        while True:
            try:
                actual_output_list.append(
                    sess.run(word_id_example_queue.get_batch()))

            except tf.errors.OutOfRangeError:
                break
"""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import collections
import tensorflow as tf

from speech.trainer.data_queue.example_queue import ExampleQueueInterface


# TODO(chanw.com) Consider implementing as a common library method.
def build_vocab(filename):
    """Builds the vocabulary from the sentences in the file.

    Args:
        filename: The file name of the text file containing multiple sentences.
            Each sentence must be separated by the new line character "\n".

    Returns:
        A dictionary containing a pair of a word and the corresponding id.
        The id is assigned based on the frequency of the words in the file
        in the decreasing order. The most frequent word will be given
        the id 1.
    """
    def read_sentence_words(filename):
        """A generator for reading a list of words in a sentence.

        "<eos>" is appended to each sentence.

        Params:
            filename: The file name of the text file containing multiple sentences.
                Each sentence must be separated by the new line character "\n".

        Returns:

        """
        with open(filename, 'rt') as in_file:
            for line in in_file:
                if line.strip():
                    words = line.rstrip().split()
                    words.insert(0, "<sos>")
                    words.append("<eos>")

                    yield words

    # Counts the words from each sentence in the file.
    counter = collections.Counter([])

    # Retrieves a list of words using the read_sentence_words generator.
    for words in read_sentence_words(filename):
        counter += collections.Counter(words)

    # Sorts pairs of (word, counts) in the decreasing order of counts.
    # If multiple words have equal counts, sorting is done based on
    # the alphabetical order of words.
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# TODO(chanw.com) Consider using the build_from_corpus method of
# tfds.features.text.SubwordTextEncoder.
# Refer to the following website.
# https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
class WordIdExampleQueue(ExampleQueueInterface):
    """An implementation of ExampleQueue for processing texts.

    Input texts are read and converted into a sequence of word ids. <sos> and
    <eos> are prepended and appended to the sentence respectively.

    Note that word IDs starts from one rather than zero. The reason is to use
    zero for zero-padding.
    """
    def __init__(self, config):
        """Constructs the WordIdExampleQueue object.

        Args:
            config: WordIdExampleQueueConfig object for initialization.

        Returns:
            A constructed WordIdExampleQueue object.
        """
        self._config = config

        # The structure of the input pipeline using tf.data is as follows:
        #
        # 1. Creates a tf.data.Dataset object.
        # 2. Applies shuffling.
        # 3. Skips blank lines.
        # 4. Prepends <sos> and appends <eos> to each utterance.
        # 5. Splits a sentence into words.
        # 6. Maps words into word IDs.
        # 7. Constructs a padded batch.

        file_name = self._config.file_name

        # 1. Creates a tf.data.Dataset object.
        dataset = tf.data.TextLineDataset(file_name)

        # 2. Applies shuffling if the randomization option is enabled.
        if self._config.randomize_order:
            dataset = dataset.shuffle(self._config.shuffle_buffer_size)

        # 3. Skips blank lines by filtering out zero-length utterances.
        dataset = dataset.filter(
            lambda line: tf.greater(
                tf.strings.length(tf.strings.strip(line)), 0))

        # 4. Prepends "<sos> " and appends " <eos>" to each sentence.
        #
        # A white space is added after <sos> or before <eos> to mark the
        # word boundaries. Note that white spaces are used for splitting
        # in the step 5.
        dataset = dataset.map(lambda line: tf.strings.join(["<sos> ", line]))
        dataset = dataset.map(lambda line: tf.strings.join([line, " <eos>"]))

        # 5. Splits a sentence into words.
        dataset = dataset.map(
            lambda line: tf.strings.split([line]).values)

        # 6. Converts words into word IDs.
        word_to_id = build_vocab(file_name)
        table_initializer = tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(word_to_id.keys())),
            tf.constant(list(word_to_id.values()), dtype=tf.int64))
        self._table = tf.lookup.StaticVocabularyTable(
            table_initializer, num_oov_buckets=1)
        dataset = dataset.map(lambda line: self._table.lookup(line) + 1)

	# 7. Performs padded batching.
        dataset = dataset.padded_batch(
            self._config.batch_size,
            padded_shapes=tf.TensorShape([None]),
            drop_remainder=True)

        self._dataset = dataset
        self._iterator = dataset.make_initializable_iterator()
        self._initializer = (
            self._iterator.initializer, self._table.initializer)


    @property
    def dataset(self):
        """Returns the dataset."""
        return self._dataset


    @property
    def initializer(self):
        """Returns the initializer."""
        return self._initializer


    def get_batch(self):
        """Returns the next batch.

        A batch is a sequence of word IDs from a sentence. If the numbers of
        words in sentences are different in a single batch, zero-padding is
        done to make the dimension the same.

        Args:
            None.

        Returns:
            A tensor object containing the next batch. A batch contains
            a sequence of word indices.
        """

        return self._iterator.get_next()
