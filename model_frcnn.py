# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import (
    Conv2D, FullyConnected, layer_register)
from tensorpack.utils.argtools import memoized

from basemodel import GroupNorm
from utils.box_ops import pairwise_iou
from model_box import encode_bbox_target, encode_mask_target, decode_bbox_target
from config import config as cfg


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, gt_masks):
    """
    Sample some ROIs from all proposals for training.
    #fg is guaranteed to be > 0, because grount truth boxes are added as RoIs.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t int64 labels, in [0, #class-1]. Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)

    ret_masks = tf.gather(gt_masks, fg_inds_wrt_gt)

    # stop the gradient -- they are meant to be training targets
    return tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'), \
        tf.stop_gradient(ret_labels, name='sampled_labels'), \
        tf.stop_gradient(fg_inds_wrt_gt), \
        tf.stop_gradient(ret_masks, name='sampled_masks')


@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class x 4)
    """
    classification = FullyConnected('class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    box_regression = FullyConnected('box', feature, num_classes * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes, 4), name='output_box')
    mask_regression = FullyConnected('mask', feature, num_classes * 5,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    mask_regression = tf.reshape(mask_regression, (-1, num_classes, 5), name='output_mask')
    return classification, box_regression, mask_regression


@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits, fg_masks, mask_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgxCx4
        fg_masks: nfgx5, encoded
        mask_logits: nfgxCx5

    Returns:
        label_loss, box_loss, mask_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    indices = tf.stack([tf.range(num_fg), fg_labels], axis=1)  # #fgx2
    fg_box_logits = tf.gather_nd(fg_box_logits, indices)
    mask_logits = tf.gather_nd(mask_logits, indices)

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int64(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.truediv(num_zero, num_fg, name='false_negative')
        fg_accuracy = tf.reduce_mean(tf.gather(correct, fg_inds), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    mask_loss = tf.losses.huber_loss(fg_masks, mask_logits, reduction=tf.losses.Reduction.SUM)
    mask_loss = tf.truediv(mask_loss, tf.to_float(tf.shape(labels)[0]), name='mask_loss')

    add_moving_summary(label_loss, box_loss, mask_loss, accuracy, fg_accuracy, false_negative)
    return label_loss, box_loss, mask_loss


@under_name_scope()
def fastrcnn_predictions(boxes, probs):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#classx4 floatbox in float32
        probs: nx#class

    Returns:
        indices: Kx2. Each is (box_id, class_id)
        probs: K floats
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert probs.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    boxes.set_shape([None, cfg.DATA.NUM_CATEGORY, None])
    probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                      parallel_iterations=10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    probs = tf.boolean_mask(probs, masks)

    # filter again by sorting scores
    topk_probs, topk_indices = tf.nn.top_k(
        probs,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(probs)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)
    final_ids = tf.stack([box_ids, cat_ids + 1], axis=1, name='final_ids')  # Kx2, each is (box_id, class_id)
    return final_ids, topk_probs


"""
FastRCNN heads for FPN:
"""


@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_classes, num_convs, norm=None):
    """
    Args:
        feature (NCHW):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'

    Returns:
        outputs of `fastrcnn_outputs()`
    """
    assert norm in [None, 'GN'], norm
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.FRCNN_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, cfg.FPN.FRCNN_FC_HEAD_DIM,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
    return fastrcnn_outputs('outputs', l, num_classes)


def fastrcnn_4conv1fc_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


def fastrcnn_4conv1fc_gn_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, norm='GN', **kwargs)


class FastRCNNHead(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """
    def __init__(self, input_boxes, input_masks, box_logits, mask_logits, label_logits, 
                labels=None, matched_gt_boxes_per_fg=None, matched_gt_masks_per_fg=None):
        """
        Args:
            input_boxes: Nx4, inputs to the head
            box_logits: Nx#classx4, the output of the head
            label_logits: Nx#class, the output of the head
            bbox_regression_weights: a 4 element tensor
            labels: N, each in [0, #class-1], the true label for each input box
            matched_gt_boxes_per_fg: #fgx4, the matching gt boxes for each fg input box

        The last two arguments could be None when not training.
        """
        self.input_boxes = input_boxes
        self.input_masks = input_masks
        self.box_logits = box_logits
        self.mask_logits = mask_logits
        self.label_logits = label_logits
        self.bbox_regression_weights = tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)
        self.mask_regression_weights = tf.constant(cfg.FRCNN.MASK_REG_WEIGHTS, dtype=tf.float32)
        self.labels = labels
        self.matched_gt_boxes_per_fg = matched_gt_boxes_per_fg
        self.matched_gt_masks_per_fg = matched_gt_masks_per_fg

    @memoized
    def fg_inds_in_inputs(self):
        """ Returns: #fg indices in [0, N-1] """
        return tf.reshape(tf.where(self.labels > 0), [-1], name='fg_inds_in_inputs')

    @memoized
    def fg_input_boxes(self):
        """ Returns: #fgx4 """
        return tf.gather(self.input_boxes, self.fg_inds_in_inputs(), name='fg_input_boxes')

    @memoized
    def fg_box_logits(self):
        """ Returns: #fg x #class x 4 """
        return tf.gather(self.box_logits, self.fg_inds_in_inputs(), name='fg_box_logits')

    @memoized
    def fg_input_masks(self):
        """ Returns: #fgx4 """
        return tf.gather(self.input_masks, self.fg_inds_in_inputs(), name='fg_input_masks')

    @memoized
    def fg_mask_logits(self):
        """ Returns: #fg x #class x 4 """
        return tf.gather(self.mask_logits, self.fg_inds_in_inputs(), name='fg_mask_logits')

    @memoized
    def fg_labels(self):
        """ Returns: #fg """
        return tf.gather(self.labels, self.fg_inds_in_inputs(), name='fg_labels')

    @memoized
    def losses(self):
        encoded_fg_gt_boxes = encode_bbox_target(self.matched_gt_boxes_per_fg,
            self.fg_input_boxes()) * self.bbox_regression_weights
        encoded_fg_gt_masks = encode_mask_target(self.matched_gt_masks_per_fg,
            self.fg_input_masks()) * self.mask_regression_weights
        return fastrcnn_losses(self.labels, self.label_logits, encoded_fg_gt_boxes, 
            self.fg_box_logits(), encoded_fg_gt_masks, self.fg_mask_logits())

    @memoized
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.input_boxes, 1),
                          [1, cfg.DATA.NUM_CLASS, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights,
            anchors
        )
        return decoded_boxes

    @memoized
    def decoded_output_boxes_for_true_label(self):
        """ Returns: Nx4 decoded boxes """
        indices = tf.stack([
            tf.range(tf.size(self.labels, out_type=tf.int64)),
            self.labels
        ])
        needed_logits = tf.gather_nd(self.box_logits, indices)
        decoded = decode_bbox_target(
            needed_logits / self.bbox_regression_weights,
            self.input_boxes
        )
        return decoded

    @memoized
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)
