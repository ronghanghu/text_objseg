from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform

from util import im_processing, text_processing, eval_tools
from models import processing_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
query_file = './exp-referit/data/referit_query_trainval.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

# Saving directory
data_folder = './exp-referit/data/train_batch_det/'
data_prefix = 'referit_train_det'

# Sample selection params
pos_iou = .7
neg_iou = 1e-6
neg_to_pos_ratio = 1.0

# Model Param
N = 50
T = 20

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
# Load training data
################################################################################

# Generate a list of training samples
# Each training sample is a tuple of 5 elements of
# (imname, imsize, sample_bbox, description, label)
# 1 as positive label and 0 as negative label (i.e. the probability of being pos)

# Gather training sample per image
# Positive training sample includes the ground-truth
training_samples_pos = []
training_samples_neg = []
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    imsize = imsize_dict[imname]
    bbox_proposals = bbox_proposal_dict[imname]
    # for each ground-truth annotation, use gt box and proposal boxes as positive examples
    # and proposal box with small iou as negative examples
    for imcrop_name in this_imcrop_names:
        if not imcrop_name in query_dict:
            continue
        gt_bbox = np.array(bbox_dict[imcrop_name]).reshape((1, 4))
        IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)
        pos_boxes = bbox_proposals[IoUs >= pos_iou, :]
        pos_boxes = np.concatenate((gt_bbox, pos_boxes), axis=0)
        neg_boxes = bbox_proposals[IoUs <  neg_iou, :]

        this_descriptions = query_dict[imcrop_name]
        # generate them per discription
        for description in this_descriptions:
            # Positive training samples
            for n_pos in range(pos_boxes.shape[0]):
                sample = (imname, imsize, pos_boxes[n_pos], description, 1)
                training_samples_pos.append(sample)
            # Negative training samples
            for n_neg in range(neg_boxes.shape[0]):
                sample = (imname, imsize, neg_boxes[n_neg], description, 0)
                training_samples_neg.append(sample)

# Print numbers of positive and negative samples
print('#pos=', len(training_samples_pos))
print('#neg=', len(training_samples_neg))

# Subsample negative training data
np.random.seed(3)
sample_idx = np.random.choice(len(training_samples_neg),
                              min(len(training_samples_neg),
                                  int(neg_to_pos_ratio*len(training_samples_pos))),
                              replace=False)
training_samples_neg_subsample = [training_samples_neg[n] for n in sample_idx]
print('#neg_subsample=', len(training_samples_neg_subsample))

# Merge and shuffle training examples
training_samples = training_samples_pos + training_samples_neg_subsample
np.random.seed(3)
perm_idx = np.random.permutation(len(training_samples))
shuffled_training_samples = [training_samples[n] for n in perm_idx]
del training_samples
print('#total sample=', len(shuffled_training_samples))

num_batch = len(shuffled_training_samples) // N
print('total batch number: %d' % num_batch)

################################################################################
# Save training samples to disk
################################################################################

text_seq_batch = np.zeros((T, N), dtype=np.int32)
imcrop_batch = np.zeros((N, 224, 224, 3), dtype=np.uint8)
spatial_batch = np.zeros((N, 8), dtype=np.float32)
label_batch = np.zeros((N, 1), dtype=np.bool)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        imname, imsize, sample_bbox, description, label = shuffled_training_samples[n_sample]
        im = skimage.io.imread(image_dir + imname)
        xmin, ymin, xmax, ymax = sample_bbox

        imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
        imcrop = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [224, 224]))
        spatial_feat = processing_tools.spatial_feature_from_bbox(sample_bbox, imsize)
        text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)

        idx = n_sample - batch_begin
        text_seq_batch[:, idx] = text_seq
        imcrop_batch[idx, ...] = imcrop
        spatial_batch[idx, ...] = spatial_feat
        label_batch[idx] = label

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
        text_seq_batch=text_seq_batch,
        imcrop_batch=imcrop_batch,
        spatial_batch=spatial_batch,
        label_batch=label_batch)
