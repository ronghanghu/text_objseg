from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from models import vgg_net, lstm_net, processing_tools
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import im_processing, text_processing, eval_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
query_file = './exp-referit/data/referit_query_test.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

pretrained_model = './exp-referit/tfmodel/referit_fc8_det_iter_25000.tfmodel'

# Model Params
T = 20
N = 100
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim

# Evaluation Param
correct_iou_thresh = 0.5
use_nms = False
nms_thresh = 0.3

################################################################################
# Evaluation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, 1])  # one batch per sentence
imcrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
lstm_top_batch = tf.placeholder(tf.float32, [N, D_text])
fc8_crop_batch = tf.placeholder(tf.float32, [N, D_im])
spatial_batch = tf.placeholder(tf.float32, [N, 8])

# Language feature (LSTM hidden state)
lstm_top = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

# Local image feature
fc8_crop = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=False)

# L2-normalize the features (except for spatial_batch)
# and concatenate them along axis 1 (feature dimension)
feat_all = tf.concat(1, [tf.nn.l2_normalize(lstm_top_batch, 1),
                         tf.nn.l2_normalize(fc8_crop_batch, 1),
                         spatial_batch])

# Outputs
# MLP Classifier over concatenate feature
with tf.variable_scope('classifier'):
    mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
    mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)
scores = mlp_l2

# Load pretrained model
snapshot_saver = tf.train.Saver()
sess = tf.Session()
snapshot_saver.restore(sess, pretrained_model)

################################################################################
# Load annotations and bounding box proposals
################################################################################

query_dict = json.load(open(query_file))
bbox_dict = json.load(open(bbox_file))
imcrop_dict = json.load(open(imcrop_file))
imsize_dict = json.load(open(imsize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

# Object proposals
bbox_proposal_dict = {}
for imname in imlist:
    bboxes = np.loadtxt(bbox_proposal_dir + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
    bbox_proposal_dict[imname] = bboxes

################################################################################
# Flatten the annotations
################################################################################

flat_query_dict = {imname: [] for imname in imlist}
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    for imcrop_name in this_imcrop_names:
        gt_bbox = bbox_dict[imcrop_name]
        if imcrop_name not in query_dict:
            continue
        this_descriptions = query_dict[imcrop_name]
        for description in this_descriptions:
            flat_query_dict[imname].append((imcrop_name, gt_bbox, description))

################################################################################
# Testing
################################################################################

eval_bbox_num_list = [1, 10, 100]
bbox_correct = np.zeros(len(eval_bbox_num_list), dtype=np.int32)
bbox_total = 0

# Pre-allocate arrays
imcrop_val = np.zeros((N, 224, 224, 3), dtype=np.float32)
spatial_val = np.zeros((N, 8), dtype=np.float32)
text_seq_val = np.zeros((T, 1), dtype=np.int32)
lstm_top_val = np.zeros((N, D_text))

num_im = len(imlist)
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    imsize = imsize_dict[imname]
    bbox_proposals = bbox_proposal_dict[imname]
    num_proposal = bbox_proposals.shape[0]
    assert(N >= num_proposal)

    # Extract visual features from all proposals
    im = skimage.io.imread(image_dir + imname)
    if im.ndim == 2:
        im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
    imcrop_val[:num_proposal, ...] = im_processing.crop_bboxes_subtract_mean(
        im, bbox_proposals, 224, vgg_net.channel_mean)
    fc8_crop_val = sess.run(fc8_crop, feed_dict={imcrop_batch:imcrop_val})

    # Extract bounding box features from proposals
    spatial_val[:num_proposal, ...] = \
        processing_tools.spatial_feature_from_bbox(bbox_proposals, imsize)

    # Extract textual features from sentences
    for imcrop_name, gt_bbox, description in flat_query_dict[imname]:
        proposal_IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)

        # Extract language feature
        text_seq_val[:, 0] = text_processing.preprocess_sentence(description, vocab_dict, T)
        lstm_top_val[...] = sess.run(lstm_top, feed_dict={text_seq_batch:text_seq_val})

        # Compute scores per proposal
        scores_val = sess.run(scores,
            feed_dict={
                lstm_top_batch:lstm_top_val,
                fc8_crop_batch:fc8_crop_val,
                spatial_batch:spatial_val
            })
        scores_val = scores_val[:num_proposal, ...].reshape(-1)

        # Sort the scores for the proposals
        if use_nms:
            top_ids = eval_tools.nms(proposal.astype(np.float32), scores_val, nms_thresh)
        else:
            top_ids = np.argsort(scores_val)[::-1]

        # Evaluate on bounding boxes
        for n_eval_num in range(len(eval_bbox_num_list)):
            eval_bbox_num = eval_bbox_num_list[n_eval_num]
            bbox_correct[n_eval_num] += \
                np.any(proposal_IoUs[top_ids[:eval_bbox_num]] >= correct_iou_thresh)
        bbox_total += 1

print('Final results on the whole test set')
result_str = ''
for n_eval_num in range(len(eval_bbox_num_list)):
    result_str += 'recall@%s = %f\n' % \
        (str(eval_bbox_num_list[n_eval_num]), bbox_correct[n_eval_num]/bbox_total)
print(result_str)
