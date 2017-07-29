from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def weighed_logistic_loss(scores, labels, pos_loss_mult=1.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    loss_mult = tf.add(tf.multiply(labels, pos_loss_mult-neg_loss_mult), neg_loss_mult)

    # Classification loss as the average of weighed per-score loss
    cls_loss = tf.reduce_mean(tf.multiply(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels),
        loss_mult))

    return cls_loss

def l2_regularization_loss(variables, weight_decay):
    l2_losses = [tf.nn.l2_loss(var) for var in variables]
    total_l2_loss = weight_decay * tf.add_n(l2_losses)
    return total_l2_loss
