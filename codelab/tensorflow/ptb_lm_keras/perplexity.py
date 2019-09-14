#!/usr/bin/python3



tf.contrib.seq2seq.sequence_loss(
    logits,
    targets,
    weights,
    average_across_timesteps=True,
    average_across_batch=True,
    sum_over_timesteps=False,
    sum_over_batch=False,
    softmax_loss_function=None,
    name=None
)


