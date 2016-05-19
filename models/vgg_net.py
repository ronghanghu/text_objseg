from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def vgg_pool5(input_batch, name):
    with tf.variable_scope(name):
        # layer 1
        conv1_1 = conv_relu('conv1_1', input_batch,
                            kernel_size=3, stride=1, output_dim=64)
        conv1_2 = conv_relu('conv1_2', conv1_1,
                            kernel_size=3, stride=1, output_dim=64)
        pool1 = pool('pool1', conv1_2, kernel_size=2, stride=2)
        # layer 2
        conv2_1 = conv_relu('conv2_1', pool1,
                            kernel_size=3, stride=1, output_dim=128)
        conv2_2 = conv_relu('conv2_2', conv2_1,
                            kernel_size=3, stride=1, output_dim=128)
        pool2 = pool('pool2', conv2_2, kernel_size=2, stride=2)
        # layer 3
        conv3_1 = conv_relu('conv3_1', pool2,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_2 = conv_relu('conv3_2', conv3_1,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_3 = conv_relu('conv3_3', conv3_2,
                            kernel_size=3, stride=1, output_dim=256)
        pool3 = pool('pool3', conv3_3, kernel_size=2, stride=2)
        # layer 4
        conv4_1 = conv_relu('conv4_1', pool3,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_2 = conv_relu('conv4_2', conv4_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_3 = conv_relu('conv4_3', conv4_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool4 = pool('pool4', conv4_3, kernel_size=2, stride=2)
        # layer 5
        conv5_1 = conv_relu('conv5_1', pool4,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_2 = conv_relu('conv5_2', conv5_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_3 = conv_relu('conv5_3', conv5_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool5 = pool('pool5', conv5_3, kernel_size=2, stride=2)
        return pool5

def vgg_fc7(input_batch, name, apply_dropout):
    pool5 = vgg_pool5(input_batch, name)
    with tf.variable_scope(name):
        # layer 6
        fc6 = fc_relu('fc6', pool5, output_dim=4096)
        if apply_dropout: fc6 = drop(fc6, 0.5)
        # layer 7
        fc7 = fc_relu('fc7', fc6, output_dim=4096)
        if apply_dropout: fc7 = drop(fc7, 0.5)
        return fc7

def vgg_fc8(input_batch, name, apply_dropout, output_dim=1000):
    fc7 = vgg_fc7(input_batch, name, apply_dropout)
    with tf.variable_scope(name):
        # layer 8 (no ReLU after fc8)
        fc8 = fc('fc8', fc7, output_dim=output_dim)
        return fc8

def vgg_fc7_full_conv(input_batch, name, apply_dropout):
    pool5 = vgg_pool5(input_batch, name)
    with tf.variable_scope(name):
        # layer 6
        fc6 = conv_relu('fc6', pool5, kernel_size=7, stride=1, output_dim=4096)
        if apply_dropout: fc6 = drop(fc6, 0.5)
        # layer 7
        fc7 = conv_relu('fc7', fc6, kernel_size=1, stride=1, output_dim=4096)
        if apply_dropout: fc7 = drop(fc7, 0.5)
        return fc7

def vgg_fc8_full_conv(input_batch, name, apply_dropout, output_dim=1000):
    fc7 = vgg_fc7_full_conv(input_batch, name, apply_dropout)
    with tf.variable_scope(name):
        # layer 8 (no ReLU after fc8)
        fc8 = conv('fc8', fc7, kernel_size=1, stride=1, output_dim=output_dim)
        return fc8
