#!/usr/bin/python3
'''Unit tests for wave_list_to_tfrecord.'''

# pylint: disable=import-error, no-name-in-module, invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf

from signal_processing.array_test import IsSimilarArray
from speech.feature import extract_feature
from speech.trainer.data_queue import example_queue_config_pb2
from speech.trainer.data_queue.create_test_tfrecord import GenerateCosineWave
from speech.trainer.data_queue.example_queue import UtteranceDataExampleQueue

__author__ = 'Chanwoo Kim(chanw.com@samsung.com)'

EPS = 1e-4


def _pad_zeros_to_waveforms(signal, length, num_channels):
    '''Pads zeros to the original signals.

    The input signal is resized to have the dimension of (length, num_channels).
    This dimension must be equal to or larger than the original signal
    dimension. The original signal is placed in the top left portion of this
    resized 2-D array. The remaining portion is filled with zeros.

    Params:
        signal: The original input signal.
        length: The length after zero-padding. It should be at least
            the original signal length.
        num_channels: The number of channels after zero padding. It should
            be at least the original number of channels.

    Returns:
        An numpy array object containing the zero-padded signal.
    '''
    assert signal.shape[0] <= length
    assert signal.shape[1] <= num_channels

    padded_array = np.zeros((int(length), int(num_channels)))
    padded_array[:signal.shape[0], :signal.shape[1]] = signal

    return padded_array


def _pad_zeros_to_features(feature, num_frames, num_filterbank_channels,
                           num_mic_channels):
    '''Pads zeros to the original feature.

    The input feature is resized to have the dimension of
    (num_frames, num_filterbank_channels, num_mic_channels).

    Params:
        feature: The original input feature.
        num_frames: The number of frames.
        num_filterbank_channels: The number of filterbank channels.
        num_mic_channels: The number of microphone channels.

    Returns:
        An numpy array object containing the zero-padded features.
    '''
    assert feature.shape[0] <= num_frames
    assert feature.shape[1] <= num_filterbank_channels
    assert feature.shape[2] <= num_mic_channels

    padded_array = np.zeros((int(num_frames), int(num_filterbank_channels),
                             int(num_mic_channels)))
    padded_array[:feature.shape[0], :feature.shape[1], :feature.shape[2]] = (
        feature)

    return padded_array


def _adds_16bit_quantization_effect(signal):
    '''Adds the 16 bit quantization effect.

    Params:
        signal: The original input signal.

    Returns:
        An array with 16-bit quantization effect.
    '''

    return (signal * 32768.0).astype(dtype=np.int16) / 32768.0


class ExampleQueueTest(unittest.TestCase):
    '''A class for testing classes and methods in wave_list_to_tfrecord.py.'''

    def __init__(self, *args, **kwargs):
        super(ExampleQueueTest, self).__init__(*args, **kwargs)
        self._TEST_TFRECORD = (
            'speech/trainer/data_queue/testdata/test_tfrecord-00000-of-00001')

        self._TEST_CONFIG_LIST = [
            # A list of tuples of (length_sec, number_of_channels) used in the
            # test_tfrecord-00000-of-00001.
            [1.0, 1],
            [4.0, 1],
            [1.0, 2],
            [4.0, 2],
        ]

        self._AMPLITUDE = 0.1
        self._FREQ_HZ = 100.0
        self._SAMPLING_RATE_HZ = 16000.0

    def test_batch_size_one_test(self):
        '''Tests the case when the batch size is one.
        This is the most basic case in this unit test.
        '''

        config = example_queue_config_pb2.Config()
        config.randomize_order = False
        config.tfrecord_file_name = self._TEST_TFRECORD

        example_queue = UtteranceDataExampleQueue(config)

        actual_result = []
        with tf.Session() as sess:
            while True:
                try:
                    actual_result.append(sess.run(example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # Because the number of data is four and the batch size is one,
        # get_batch() should have been called successfully four times.
        self.assertEqual(4, len(actual_result))

        # The batch size is the size of actual_result[0][0].shape[0].
        self.assertEqual(1, actual_result[0][0].shape[0])

        for i in range(4):
            # Checks the number of channels.
            self.assertEqual(self._TEST_CONFIG_LIST[i][1],
                             actual_result[i][0][0])

            # Checks the sampling rate.
            self.assertEqual(16000.0, actual_result[i][1][0])

            # Checks the waveform.
            expected_wave_data = GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[i][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[i][1])
            self.assertTrue(
                IsSimilarArray(expected_wave_data, actual_result[i][2][0],
                               EPS))

            # Checks the transcript.
            self.assertEqual(b'', actual_result[i][3][0])

    def test_batch_size_three_test(self):
        '''Tests the case when the the batch size is three.'''

        config = example_queue_config_pb2.Config()
        config.batch_size = 3
        config.randomize_order = False
        config.tfrecord_file_name = self._TEST_TFRECORD

        example_queue = UtteranceDataExampleQueue(config)

        actual_result = []
        with tf.Session() as sess:
            while True:
                try:
                    actual_result.append(sess.run(example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # Because the number of data is four and the batch size is three,
        # get_batch() should have been called successfully only once.
        self.assertEqual(1, len(actual_result))

        # The batch size is the size of actual_result[0][0].shape[0].
        self.assertEqual(config.batch_size, actual_result[0][0].shape[0])

        # Checks the number of channels.
        self.assertEqual(1, actual_result[0][0][0])
        self.assertEqual(1, actual_result[0][0][1])
        self.assertEqual(2, actual_result[0][0][2])

        # Checks the sampling rate in Hz.
        self.assertEqual(16000.0, actual_result[0][1][0])
        self.assertEqual(16000.0, actual_result[0][1][1])
        self.assertEqual(16000.0, actual_result[0][1][2])

        # Checks the wave data.
        expected_wave_data = _pad_zeros_to_waveforms(
            GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[0][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[0][1]),
            16000.0 * 4.0, 2)
        self.assertTrue(
            IsSimilarArray(expected_wave_data, actual_result[0][2][0], EPS))

        expected_wave_data = _pad_zeros_to_waveforms(
            GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[1][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[1][1]),
            16000.0 * 4.0, 2)
        self.assertTrue(
            IsSimilarArray(expected_wave_data, actual_result[0][2][1], EPS))

        expected_wave_data = _pad_zeros_to_waveforms(
            GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[2][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[2][1]),
            16000.0 * 4.0, 2)
        self.assertTrue(
            IsSimilarArray(expected_wave_data, actual_result[0][2][2], EPS))

        # Checks the transcript.
        self.assertEqual(b'', actual_result[0][3][0])
        self.assertEqual(b'', actual_result[0][3][1])
        self.assertEqual(b'', actual_result[0][3][2])

    def test_batch_size_two_max_utterance_length_test(self):
        '''Tests the case when the batch size is two and the maximum
        utterance length limit is 2.0 seconds.

        '''

        config = example_queue_config_pb2.Config()
        config.batch_size = 2
        config.randomize_order = False
        config.max_utterance_length_filter_sec = 2.0
        config.tfrecord_file_name = self._TEST_TFRECORD

        example_queue = UtteranceDataExampleQueue(config)

        actual_result = []
        with tf.Session() as sess:
            while True:
                try:
                    actual_result.append(sess.run(example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # Because the number of utterances whose length is less than two
        # seconds is two and the batch size is two, get_batch() should have
        # been called successfully only once.
        self.assertEqual(1, len(actual_result))

        # The batch size is the size of actual_result[0][0].shape[0].
        self.assertEqual(config.batch_size, actual_result[0][0].shape[0])

        # Checks the number of channels.
        self.assertEqual(1, actual_result[0][0][0])
        self.assertEqual(2, actual_result[0][0][1])

        # Checks the sampling rate in Hz.
        self.assertEqual(16000.0, actual_result[0][1][0])
        self.assertEqual(16000.0, actual_result[0][1][1])

        # Checks the wave data.
        expected_wave_data = _pad_zeros_to_waveforms(
            GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[0][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[0][1]),
            16000.0 * 1.0, 2)
        self.assertTrue(
            IsSimilarArray(expected_wave_data, actual_result[0][2][0], EPS))

        expected_wave_data = _pad_zeros_to_waveforms(
            GenerateCosineWave(
                self._AMPLITUDE, self._FREQ_HZ, self._TEST_CONFIG_LIST[2][0],
                self._SAMPLING_RATE_HZ, self._TEST_CONFIG_LIST[2][1]),
            16000.0 * 1.0, 2)
        self.assertTrue(
            IsSimilarArray(expected_wave_data, actual_result[0][2][1], EPS))

        # Checks the transcript.
        self.assertEqual(b'', actual_result[0][3][0])
        self.assertEqual(b'', actual_result[0][3][1])

    def test_batch_size_three_feat_ext_test(self):
        '''Tests the case when feature extraction is also done.'''

        config = example_queue_config_pb2.Config()
        config.batch_size = 3
        config.randomize_order = False
        config.tfrecord_file_name = self._TEST_TFRECORD

        # Sets the feature_ext_config.
        feature_ext_config = config.feature_ext_config
        feature_ext_config.feature_type = (
            example_queue_config_pb2.FeatExtConfig.POWER_MEL_FILTERBANK)

        example_queue = UtteranceDataExampleQueue(config)

        actual_result = []
        with tf.Session() as sess:
            while True:
                try:
                    actual_result.append(sess.run(example_queue.get_batch()))

                except tf.errors.OutOfRangeError:
                    break

        # Because the number of data is four and the batch size is three,
        # get_batch() should have been called successfully only once.
        self.assertEqual(1, len(actual_result))

        # The batch size is the size of actual_result[0][0].shape[0].
        self.assertEqual(config.batch_size, actual_result[0][0].shape[0])

        # Checks the number of channels.
        self.assertEqual(1, actual_result[0][0][0])
        self.assertEqual(1, actual_result[0][0][1])
        self.assertEqual(2, actual_result[0][0][2])

        # Checks the sampling rate in Hz.
        self.assertEqual(16000.0, actual_result[0][1][0])
        self.assertEqual(16000.0, actual_result[0][1][1])
        self.assertEqual(16000.0, actual_result[0][1][2])

        SAMPLING_RATE_HZ = 16000.0
        FRAME_SIZE_SEC = 0.025
        FRAME_STEP_SEC = 0.010
        FEATURE_SIZE = 40
        MAX_NUM_FRAMES = 401
        MAX_NUM_MIC_CHANNELS = 2
        FEATURE_TYPE = 'power_mel_filterbank'

        # Checks the feature data.
        expected_feature_data = _pad_zeros_to_features(
            extract_feature.WaveArrayToFeatureArray(
                _adds_16bit_quantization_effect(
                    GenerateCosineWave(
                        self._AMPLITUDE, self._FREQ_HZ,
                        self._TEST_CONFIG_LIST[0][0], self._SAMPLING_RATE_HZ,
                        self._TEST_CONFIG_LIST[0][1])), SAMPLING_RATE_HZ,
                FRAME_SIZE_SEC, FRAME_STEP_SEC, FEATURE_SIZE, FEATURE_TYPE),
            MAX_NUM_FRAMES, FEATURE_SIZE, MAX_NUM_MIC_CHANNELS)
        self.assertTrue(
            IsSimilarArray(expected_feature_data, actual_result[0][2][0], EPS))

        expected_feature_data = _pad_zeros_to_features(
            extract_feature.WaveArrayToFeatureArray(
                _adds_16bit_quantization_effect(
                    GenerateCosineWave(
                        self._AMPLITUDE, self._FREQ_HZ,
                        self._TEST_CONFIG_LIST[1][0], self._SAMPLING_RATE_HZ,
                        self._TEST_CONFIG_LIST[1][1])), SAMPLING_RATE_HZ,
                FRAME_SIZE_SEC, FRAME_STEP_SEC, FEATURE_SIZE, FEATURE_TYPE),
            MAX_NUM_FRAMES, FEATURE_SIZE, MAX_NUM_MIC_CHANNELS)
        self.assertTrue(
            IsSimilarArray(expected_feature_data, actual_result[0][2][1], EPS))

        expected_feature_data = _pad_zeros_to_features(
            extract_feature.WaveArrayToFeatureArray(
                _adds_16bit_quantization_effect(
                    GenerateCosineWave(
                        self._AMPLITUDE, self._FREQ_HZ,
                        self._TEST_CONFIG_LIST[2][0], self._SAMPLING_RATE_HZ,
                        self._TEST_CONFIG_LIST[2][1])), SAMPLING_RATE_HZ,
                FRAME_SIZE_SEC, FRAME_STEP_SEC, FEATURE_SIZE, FEATURE_TYPE),
            MAX_NUM_FRAMES, FEATURE_SIZE, MAX_NUM_MIC_CHANNELS)
        self.assertTrue(
            IsSimilarArray(expected_feature_data, actual_result[0][2][2], EPS))

        # Checks the transcript.
        self.assertEqual(b'', actual_result[0][3][0])
        self.assertEqual(b'', actual_result[0][3][1])
        self.assertEqual(b'', actual_result[0][3][2])


if __name__ == '__main__':
    unittest.main()
