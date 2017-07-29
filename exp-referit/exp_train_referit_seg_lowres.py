from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np

from models import text_objseg_model as segmodel
from util import data_reader
from util import loss

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
N = 10
input_H = 512; featmap_H = (input_H // 32)
input_W = 512; featmap_W = (input_W // 32)
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

# Initialization Params
pretrained_model = './exp-referit/tfmodel/referit_fc8_seg_lowres_init.tfmodel'

# Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_lr = 0.01
lr_decay_step = 10000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.9
max_iter = 30000

fix_convnet = False
vgg_dropout = False
mlp_dropout = False
vgg_lr_mult = 1.

# Data Params
data_folder = './exp-referit/data/train_batch_seg/'
data_prefix = 'referit_train_seg'

# Snapshot Params
snapshot = 5000
snapshot_file = './exp-referit/tfmodel/referit_fc8_seg_lowres_iter_%d.tfmodel'

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])
label_batch = tf.placeholder(tf.float32, [N, featmap_H, featmap_W, 1])

# Outputs
scores = segmodel.text_objseg_full_conv(text_seq_batch, imcrop_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=vgg_dropout, mlp_dropout=mlp_dropout)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################

# Only train the fc layers of convnet and keep conv layers fixed
if fix_convnet:
    train_var_list = [var for var in tf.trainable_variables()
                      if not var.name.startswith('vgg_local/')]
else:
    train_var_list = [var for var in tf.trainable_variables()
                      if not var.name.startswith('vgg_local/conv')]
print('Collecting variables to train:')
for var in train_var_list: print('\t%s' % var.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
reg_var_list = [var for var in tf.trainable_variables()
                if (var in train_var_list) and
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.name)
print('Done.')

# Collect learning rate for trainable variables
var_lr_mult = {var: (vgg_lr_mult if var.name.startswith('vgg_local') else 1.0)
               for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

cls_loss = loss.weighed_logistic_loss(scores, label_batch, pos_loss_mult, neg_loss_mult)
reg_loss = loss.l2_regularization_loss(reg_var_list, weight_decay)
total_loss = cls_loss + reg_loss

################################################################################
# Solver
################################################################################

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_lr, global_step, lr_decay_step,
    lr_decay_rate, staircase=True)
solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
# Compute gradients
grads_and_vars = solver.compute_gradients(total_loss, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)

################################################################################
# Initialize parameters and load data
################################################################################

snapshot_loader = tf.train.Saver(tf.trainable_variables())

# Load data
reader = data_reader.DataReader(data_folder, data_prefix)

snapshot_saver = tf.train.Saver()
sess = tf.Session()

# Run Initialization operations
sess.run(tf.global_variables_initializer())
snapshot_loader.restore(sess, pretrained_model)

################################################################################
# Optimization loop
################################################################################

cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

for n_iter in range(max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val = batch['text_seq_batch']
    imcrop_val = batch['imcrop_batch'].astype(np.float32) - segmodel.vgg_net.channel_mean
    label_val = batch['label_coarse_batch'].astype(np.float32)

    # Forward and Backward pass
    scores_val, cls_loss_val, _, lr_val = sess.run([scores, cls_loss, train_step, learning_rate],
        feed_dict={
            text_seq_batch  : text_seq_val,
            imcrop_batch    : imcrop_val,
            label_batch     : label_val
        })
    cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
    print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, lr = %f'
        % (n_iter, cls_loss_val, cls_loss_avg, lr_val))

    # Accuracy
    accuracy_all, accuracy_pos, accuracy_neg = segmodel.compute_accuracy(scores_val, label_val)
    avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
    avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
    avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
    print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
          % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
    print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
          % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) >= max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
