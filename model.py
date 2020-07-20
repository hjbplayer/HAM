#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('network', 'resnet_v1_50', '')
tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('hard_pos_samples', 128, '')
tf.app.flags.DEFINE_integer('rand_pos_samples', 128, '')
tf.app.flags.DEFINE_integer('hard_neg_samples', 512, '')
tf.app.flags.DEFINE_integer('rand_neg_samples', 512, '')
tf.app.flags.DEFINE_float('anchor_loss_weight', 0.1, 'anchor_selector_loss_weight')

from nets import nets_factory

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, anchor_sizes, select_split_N, is_select_background=False, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    base_network = nets_factory.get_network_fn(FLAGS.network, num_classes=2, weight_decay=weight_decay, is_training=is_training)
    logits, end_points = base_network(images)
    #with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
    #    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            #selector_N = 10
            #anchor_sizes = [16, 32, 64, 128, 256, 512]
            #select_thresh = 0.4
            #softmax_scale = 10.0
            selector_N = len(anchor_sizes)
            if is_select_background:
                background_channel = 1
            else:
                background_channel = 0
            
            fm_N = slim.conv2d(g[3], (selector_N + background_channel) * select_split_N, 1, activation_fn=None, normalizer_fn=None)
            #fm_clsN = slim.conv2d(g[3], (selector_N + background_channel) * select_split_N, 1, activation_fn=None, normalizer_fn=None)

            
            geo_maps_wh = []
            F_scores = []
            fm_N_split = tf.split(fm_N, select_split_N, axis=-1)
            #fm_clsN = tf.split(fm_clsN, select_split_N, axis=-1)
            for f_i in range(select_split_N):
                reg_selector = tf.nn.softmax(fm_N_split[f_i%select_split_N], axis=-1)
                reg_selector_split = tf.split(reg_selector, (selector_N + background_channel), axis=-1)
                F_score_one = reg_selector_split[0]
                for s_i in range(1, selector_N):
                    F_score_one += reg_selector_split[s_i]
                F_scores.append(F_score_one)
                geo_map_ones_wh = []
                for s_i in range(selector_N):
                    geo_map_ones_wh.append(tf.exp(slim.conv2d(g[3], 1, 1, activation_fn=None, normalizer_fn=None))
                                        * anchor_sizes[s_i] * reg_selector_split[s_i])
                geo_map_one_wh = geo_map_ones_wh[0]
                for s_i in range(1, selector_N):
                    geo_map_one_wh += geo_map_ones_wh[s_i]

                geo_maps_wh.append(geo_map_one_wh)
            
            geo_map_w_ratio = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            geo_map_h_ratio = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            geo_maps = []
            t_ = geo_maps_wh[1] * geo_map_h_ratio
            r_ = geo_maps_wh[0] * geo_map_w_ratio
            b_ = geo_maps_wh[1] * (1 - geo_map_h_ratio)
            l_ = geo_maps_wh[0] * (1 - geo_map_w_ratio)
            geo_maps.append(t_)
            geo_maps.append(r_)
            geo_maps.append(b_)
            geo_maps.append(l_)
            
            geo_map = tf.concat(geo_maps,axis=-1)

            F_score = F_scores[0]
            for s_i in range(1, 2):
                F_score += F_scores[s_i]
            F_score = F_score / 2.0

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            #F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            #geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry, fm_N

