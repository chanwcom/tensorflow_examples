#!/usr/bin/python3
'''A module for generating a TFRecord file to be used in example_queue_test.py.

Example Usage:
    bazel run :create_test_tfrecord -- --tfrecord_file_name=/tmp/test_tfrecord
'''

#pylint: disable=import-error, invalid-name, no-member, no-name-in-module

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from speech.common import utterance_data_pb2

__author__ = 'Chanwoo Kim(chanw.com@samsung.com)'


# TODO(chanw.com) Consider making the following routine as a general library
# function.
def NumChannels(numpy_array):
    rank = len(numpy_array.shape)
    if rank == 1:
        return 1
    if rank == 2:
        return numpy_array.shape[1]
    raise Exception('The rank must be one or two, but the input array has a '
                    'rank of {0}'.format(shape(numpy_array)))


# TODO(chanw.com) Consider making the following routine as a general library
# function.
def GenerateCosineWave(amplitude, freq_hz, length_sec, sampling_rate_hz,
                       number_of_channels):
    '''Generates a sine wave of frequency of freq_hz.

    Params:
        amplitude: The amplitude of the cosine wave.
        freq_hz: The frequency of the sinusoidal wave in Hertz.
        length_sec: The length of the wave in seconds.
        sampling_rate_hz: The sampling rate in Hertz.
        number_of_channels: The number of channels of the output cosine wave.

    Returns:
        A numpy array containing the generated cosine wave.
    '''

    assert freq_hz > 0.0
    assert length_sec >= 0.0
    assert sampling_rate_hz > 0.0

    length_samples = np.round(length_sec * sampling_rate_hz)

    samples = (amplitude * np.cos(2.0 * np.pi * freq_hz / sampling_rate_hz *
                                  np.arange(length_samples, dtype=np.double)))
    samples = np.tile(samples, (number_of_channels, 1)).transpose()

    return samples


# TODO(chanw.com) Consider making the following function as a general purpose
# function after making it support various types such as int8, int16, int32,
# float32, float64.
def CreateUtteranceData(sampling_rate_hz, samples, utterance_id):
    '''Creates an UtteranceData protocol message containing a cosine waveform.

    Params:
        samples: A NumPy array object containing wave data.
        utterance_id: The unique utterance ID.

    Returns:
        An UtteranceData object containing a cosine waveform.

    '''
    wave_header = utterance_data_pb2.WaveHeader()
    wave_header.number_of_channels = NumChannels(samples)
    wave_header.sampling_rate_hz = sampling_rate_hz
    wave_header.atomic_type = "int16"

    utterance_data = utterance_data_pb2.UtteranceData()

    utterance_data.utterance_id = utterance_id
    utterance_data.wave_header.CopyFrom(wave_header)
    utterance_data.samples = ((samples * 32768.0).astype(
        dtype=np.int16).tobytes())
    # Since the waveform is not active speech, the transcript is empty.
    utterance_data.ground_truth_transcript = ""

    return utterance_data


def WriteUtteranceData(tfrecord_name):
    '''Creates objects of UtteranceData and stores them to a TFRecord.

    Args:
        tfrcord_name: The TFRecord file name where objects of UtteranceData will
            be stored.

    Returns:
        None.
    '''
    AMPLITUDE = 0.1
    FREQ_HZ = 100.0
    SAMPLING_RATE_HZ = 16000.0

    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    utt_index = 0
    for number_of_channel in range(1, 3):
        for length_sec in [1.0, 4.0]:
            # The base portion of the utterance id was randomly created using
            # int(uuid.uuid4().hex, 16)
            utterance_id = '{0:x}'.format(
                21306532076565608446625527731002683816 + utt_index)
            samples = GenerateCosineWave(AMPLITUDE, FREQ_HZ, length_sec,
                                         SAMPLING_RATE_HZ, number_of_channel)
            utterance_data = CreateUtteranceData(SAMPLING_RATE_HZ, samples,
                                                 utterance_id)

            # Writes utterance_data to the TFRecord file.
            writer.write(utterance_data.SerializeToString())
            utt_index += 1

    writer.close()


def main(args):
    '''Main entry function.

    Params:
        args: Arguments obtained using argparse.

    Returns:
        None.
    '''
    WriteUtteranceData(args.tfrecord_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfrecord_file_name',
        default=None,
        help='The output TFRecord file name',
        required=True,
        type=str)

    main(parser.parse_args())
