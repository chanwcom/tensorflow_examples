#!/usr/bin/python3
"""This ExampleQueue class provides each batch from the data stored in sharded
TFRecordData.
"""

# pylint: disable=import-error, no-name-in-module, no-member
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import abc
import numpy as np
import tensorflow as tf

from speech.common import utterance_data_pb2
from speech.common.utterance_data_proto_helper import UtteranceDataProtoHelper
from speech.feature.extract_feature import WaveArrayToFeatureArray
from speech.trainer.data_queue import example_queue_config_pb2


class ExampleQueueInterface(object):
    """An example queue interface.

    This queue wraps a data reading pipeline implemented using a tf.data.
    The data set should return a tuple of (inputs, targets) to be compatible
    with the fit and the fit_generator methods of the tf.keras.Model.

    To retrieve data, the following two methods are recommended.

    1) Using the dataset property.

    Using the dataset property, we may get the internal dataset. This dataset
    my be fed into tf.keras.Model as an argument to the fit method.

    2) Using the get_batch method.

    Using the get_batch method, we may use either tf.session or the
    fit_generator method of tf.kears.Model to retrieve data.

    """

    def __init__(self, config=None):
        """Creates the object."""
        pass

    @property
    @abc.abstractmethod
    def dataset(self):
        """Returns an internal tf.data.dataset object.

        Args:
            None.

        Returns:
            A tf.data.dataset object.
        """
        pass

    @abc.abstractmethod
    def get_batch(self):
        """Returns the next batch.

        This method should be implemented as a generator returning a tuple of
        (inputs, targets). "inputs" contain batches of the input data.
        "targets" contain the corresponding target values.

        Returns:
           A tuple of (inputs, targets).
        """
        pass

    @property
    @abc.abstractmethod
    def config(self):
        """Returns the config."""
        pass

    @config.setter
    @abc.abstractmethod
    def config(self, config):
        """Sets the config."""
        pass


class UtteranceDataExampleQueue(ExampleQueueInterface):
    """A class for constructing the example queue pipeline.

    This class assumes that the input is a sharded TFRecord files.
    The configuration for setting up the pipeline is described in
    the example_queue_config.proto.

    Example usage:

        config = example_queue_config_pb2.Config()
        config.randomize_order = False
        config.tfrecord_file_name = self._TEST_TFRECORD

        example_queue = ExampleQueue(config)

        utterance_data = utterance_data_pb2.UtteranceData()

        batch = example_queue.get_batch()
    """

    def __init__(self, config=None):
        """Constructs the data pipeline graph for getting a batch.

        Args:
            config: Configuration for ExampleQueue (example_queue_config.proto)

        Returns:
            None.
        """
        self._config = config

        # Creation of the dataset from the TFRecord file name.
        dataset = (tf.data.Dataset.list_files(
            self._config.tfrecord_file_name,
            shuffle=bool(self._config.randomize_order)).apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    cycle_length=self._config.num_parallel_calls,
                    sloppy=self._config.randomize_order)))

        # Applies shuffling if the randomization option is enabled.
        if self._config.randomize_order:
            dataset = dataset.shuffle(self._config.shuffle_buffer_size)

        # Parses UtteranceData using tf.data.Dataset.map().
        #
        # The output format is (num_channels, sampling_rate_hz, wave_data,
        # transcript).
        dataset = dataset.map(
            lambda serialized_data: tf.py_func(self._parse_utterance_data, [
                serialized_data
            ], (tf.int32, tf.float64, tf.float64, tf.string)),
            num_parallel_calls=self._config.num_parallel_calls)

        # Filters data by the utterance length.
        dataset = dataset.filter(self._filter_by_utterance_length)

        # Performs feature extraction using dataset.map.
        if self._config.HasField("feature_ext_config"):
            dataset = dataset.map(
                lambda num_channels, sampling_rate_hz, wave_data, transcript:
                tf.py_func(self._feature_ext, [
                    num_channels, sampling_rate_hz, wave_data, transcript
                ], (tf.int32, tf.float64, tf.float64, tf.string)),
                num_parallel_calls=self._config.num_parallel_calls)

        # Performs padded batch.
        #
        # For features, the rank is three as shown below:
        # (number_of_frames x number_of_filterbank_channels, num_microphones).
        if self._config.HasField("feature_ext_config"):
            # Performs padded batching.
            padded_shapes = (tf.TensorShape([]), tf.TensorShape([]),
                             tf.TensorShape([None, None, None]),
                             tf.TensorShape([]))

        # for waveforms, the rank is two as shown below:
        # (number_of_samples, num_microphones).
        else:
            # Performs padded batching.
            padded_shapes = (tf.TensorShape([]), tf.TensorShape([]),
                             tf.TensorShape([None, None]), tf.TensorShape([]))

        dataset = dataset.padded_batch(
            self._config.batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
        self._next = iterator.get_next()
        super(UtteranceDataExampleQueue, self).__init__()

    def get_batch(self):
        """Returns the next batch.

        A batch is a tuple of four elements:
        (number_of_channels, sampling_rate_hz, wave_data, transcript).
        When the batch size is more than one, and if the number of
        channels and the lengths of signals are different within a single
        batch, zero-padding is done to make the dimension the same.

        Args:
            None.

        Returns:
            A tensor object containing the next batch.

            Refers to _parse_utterance_data about the parsing procedure.
        """
        assert self._config

        return self._next


    @property
    def config(self):
        """Returns config."""
        return self._config


    @config.setter
    def config(self, config):
        """Sets the config."""
        self._config = config


    def _parse_utterance_data(self, serialized_buffer):
        """Parses a serialized buffer containing the UtteranceData message.

        Args:
            serialized_buffer: A string containing an UtteranceData value.

        Returns:
            A tuple of four elements:
            (number_of_channels, sampling_rate_hz, wave_data, transcript).
        """

        utterance_data = utterance_data_pb2.UtteranceData()
        utterance_data.ParseFromString(serialized_buffer)
        utterance_data_helper = UtteranceDataProtoHelper(utterance_data)
        (sampling_rate_hz, wave_data) = utterance_data_helper.read_wave_data()
        transcript = utterance_data_helper.read_transcript()
        num_channels = np.int32(wave_data.shape[1])

        return (num_channels, sampling_rate_hz, wave_data, transcript)


    def _feature_ext(self, num_channels, sampling_rate_hz, wave_data,
                     transcript):
        """Performs feature extraction.

        Args:
            num_channels: The number of channels of wave_data.
            sampling_rate_hz: The sampling rate in Hz of wave_data.
            wave_data: The waveform data. It must be a Tensor of rank two.
            transcript: The training transcript.

        Returns:
            A tuple of four objects:
            (number_of_channels, sampling_rate_hz, feature_data, transcript).
        """
        assert self._config.HasField("feature_ext_config")

        # TODO(chanw.com)
        # The following feature type conversion is not elegant.
        # Replace the string based feature type in extract_feature.py
        # with FeatExtConfig in example_queue_config.proto.
        feature_type = self._config.feature_ext_config.feature_type
        if (feature_type ==
                example_queue_config_pb2.FeatExtConfig.POWER_MEL_FILTERBANK):
            feature_string = "power_mel_filterbank"
        elif feature_type == example_queue_config_pb2.FeatExtConfig.PCC:
            feature_string = "pcc"
        elif (feature_type ==
              example_queue_config_pb2.FeatExtConfig.MEL_FILTERBANK):
            feature_string = "mel_filterbank"
        elif (feature_type ==
              example_queue_config_pb2.FeatExtConfig.LOG_MEL_FILTERBANK):
            feature_string = "log_mel_filterbank"
        elif (feature_type ==
              example_queue_config_pb2.FeatExtConfig.GAMMATONE_FILTERBANK):
            feature_string = "gammatone_filterbank"
        else:
            raise Exception("Unsupported type {0}".format(feature_type))

        feature_data = WaveArrayToFeatureArray(
            wave_data, sampling_rate_hz,
            self._config.feature_ext_config.frame_size_sec,
            self._config.feature_ext_config.frame_step_sec,
            self._config.feature_ext_config.feature_size, feature_string,
            self._config.feature_ext_config.log_floor)

        return (num_channels, sampling_rate_hz, feature_data, transcript)


    def _filter_by_utterance_length(self, *parsed_data):
        """Filters using the max_utterance_length_filter_sec field.

        If the utterance length is larger than the
        max_utterance_length_filter_sec field of the Config protocol buffer,
        that utterance is filtered out from the data pipeline.

        Args:
            parsed_data: A tuple of the following:
                (number_of_channels, sampling_rate_hz, wave_data, transcript)

        Returns:
            True if the utterance length is equal to or larger than the
                 max_utterance_length_filter_sec field of the config protocol
                 buffer.
        """

        # The length of the utterance is given by the length of wave data
        # divided by the sampling rate in Hz.
        # Note that the length of the wave data is tf.shape(parsed_data[2])[0],
        # and the sampling rate is parsed_data[1].
        # Refer to the _parse_utterance_data method about how parsing is done.
        return tf.math.greater_equal(
            tf.dtypes.cast(
                self._config.max_utterance_length_filter_sec,
                dtype=tf.float64),
            tf.divide(
                # The length of the wave data.
                tf.dtypes.cast(tf.shape(parsed_data[2])[0], dtype=tf.float64),
                # The sampling rate in Hz.
                parsed_data[1]))
