from __future__ import absolute_import, division, print_function

import numpy as np
import pyximport; pyximport.install()
from util.nms import cpu_nms as nms

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_bbox_iou(bboxes, target):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    if isinstance(target, list):
        target = np.array(target)
    target = target.reshape((-1, 4))

    A_bboxes = (bboxes[..., 2]-bboxes[..., 0]+1) * (bboxes[..., 3]-bboxes[..., 1]+1)
    A_target = (target[..., 2]-target[..., 0]+1) * (target[..., 3]-target[..., 1]+1)
    assert(np.all(A_bboxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(bboxes[..., 0], target[..., 0])
    I_y1 = np.maximum(bboxes[..., 1], target[..., 1])
    I_x2 = np.minimum(bboxes[..., 2], target[..., 2])
    I_y2 = np.minimum(bboxes[..., 3], target[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U
