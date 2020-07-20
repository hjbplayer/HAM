#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_boolean('select_wh', True, 'True: use w,h; False:use t,r,b,l')
tf.app.flags.DEFINE_boolean('is_select_background', True, 'select_background')
tf.app.flags.DEFINE_float('resize_ratio', 1.0, '')
tf.app.flags.DEFINE_integer('max_side_len', 2400, '')
tf.app.flags.DEFINE_float('threshold', 0.9, '')
tf.app.flags.DEFINE_float('box_thresh', 0.1, '')
tf.app.flags.DEFINE_boolean('IRB', True, '')
tf.app.flags.DEFINE_integer('start_IRB_max_len', 0, '')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS
ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
if FLAGS.select_wh:
    select_split_N = 2
else:
    select_split_N = 4

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=FLAGS.max_side_len, re_ratio=1.):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if re_ratio < 0:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    if max(resize_h * re_ratio, resize_w * re_ratio) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = re_ratio
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def shrink_poly_hjb_v0(poly, r, shrink_ratio=0.3):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = shrink_ratio
    sr_ = (0.5 - shrink_ratio) / 0.5
    point_center = [0,0]
    if len(poly) != 4:
        print("len(poly) != 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None
    for i in range(len(poly)):
        point_center[0] += poly[i][0]/len(poly)
        point_center[1] += poly[i][1]/len(poly)
        
    for i in range(len(poly)):
        p_mid = [poly[i][0] - point_center[0], poly[i][1] - point_center[1]]
        poly[i][0] = point_center[0] + p_mid[0] * sr_
        poly[i][1] = point_center[1] + p_mid[1] * sr_
    
    return poly

def shrink_poly(poly, r, shrink_ratio=0.3):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = shrink_ratio
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly

def calc_iou(poly1, poly2):
    size1 = cv2.contourArea((poly1[0:8].reshape((4,2))).astype(np.float32))
    size2 = cv2.contourArea((poly2[0:8].reshape((4,2))).astype(np.float32))
    inter = cv2.intersectConvexConvex((poly1[0:8].reshape((4,2))).astype(np.float32), (poly2[0:8].reshape((4,2))).astype(np.float32))
    inter_size = inter[0]
    if size1 + size2 - inter_size == 0:
        print("calc_iou error, size1 + size2 - inter_size == 0 !!!!!!!!!!!!")
        return 0
    iou = inter_size / (size1 + size2 - inter_size)
    return iou

#iterative regression box
def IRB(box,score_map, geo_map,score_map_thresh,show_log=True,iter_max=10,iter_stop_iou=0.99, merge_iou=0.2):
    #########################IRB#########################
    #->:all box -> nms -> box 
    #1-> in box,4points
    #2->4box
    #3-> min area box
    start_time = time.time()
    pre_box = box
    iou = 0
    pre_iou = -1
    iter_cnt = 0
    #point_boxes = []
    while iou < iter_stop_iou and pre_iou != iou and iter_cnt < iter_max:
        pre_iou = iou
        iter_cnt += 1
        #####1-> in box,4points:   p0-p3
        b = pre_box // 4
        min1 =  max(min(b[0],b[2],b[4],b[6]), 0)
        max1 =  min(max(b[0],b[2],b[4],b[6]), score_map.shape[1])
        min2 =  max(min(b[1],b[3],b[5],b[7]), 0)
        max2 =  min(max(b[1],b[3],b[5],b[7]), score_map.shape[0])
        #local_score = score_map[int(min1//4) : int(max1//4+1), int(min2//4) : int(max2//4+1)]
        local_score = score_map[int(min2) : int(max2+1), int(min1) : int(max1+1)]
        local_b = np.array([b[0]-min1, b[1]-min2, b[2]-min1, b[3]-min2, b[4]-min1, b[5]-min2, b[6]-min1, b[7]-min2])
        local_score = np.array(local_score)
        mask = np.zeros_like(local_score, dtype=np.uint8)
        
        shrinked = True
        if shrinked:
            poly = local_b.reshape((4, 2)).astype(np.int32)
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            shrinked_poly = shrink_poly_hjb_v0(poly.copy(), r,shrink_ratio=0.3)
            #shrinked_poly = shrink_poly(poly.copy(), r,shrink_ratio=0.3)
            cv2.fillPoly(mask, shrinked_poly.astype(np.int32)[np.newaxis, :, :], 1)
        else:
            cv2.fillPoly(mask, local_b.reshape((-1, 4, 2)).astype(np.int32), 1)
            
        local_score_masked = local_score * mask
        
        xy_text = np.argwhere(local_score_masked > score_map_thresh)
        if len(xy_text) == 0:
            if shrinked == False:
                return pre_box, time.time()-start_time, iter_cnt
            else:
                cv2.fillPoly(mask, local_b.reshape((-1, 4, 2)).astype(np.int32), 1)
                local_score_masked = local_score * mask
                xy_text = np.argwhere(local_score_masked > score_map_thresh)
                if len(xy_text) == 0:
                    return pre_box, time.time()-start_time, iter_cnt
        p0 = np.argmin(xy_text[:,0])
        p1 = np.argmax(xy_text[:,0])
        p2 = np.argmin(xy_text[:,1])
        p3 = np.argmax(xy_text[:,1])
        
        #####2->4box:    b_s[]
        mask = np.zeros_like(local_score_masked, dtype=np.uint8)
        mask[xy_text[p0,:][0],xy_text[p0,:][1]] = 1
        mask[xy_text[p1,:][0],xy_text[p1,:][1]] = 1
        mask[xy_text[p2,:][0],xy_text[p2,:][1]] = 1
        mask[xy_text[p3,:][0],xy_text[p3,:][1]] = 1
        
        xy_text = np.argwhere(mask == 1)
        xy_text[:,0] += int(min2)
        xy_text[:,1] += int(min1)
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        #print("xy_text:",xy_text,len(xy_text)," points:",p0,p1,p2,p3)
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        b_s = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        b_s[:, :8] = text_box_restored.reshape((-1, 8))
        b_s[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        
        
        
        #####3-> min area box
        score_points = b_s[:, 8].sum() / len(b_s)
        score = score_points
        b_iou = []
        for pb in b_s:
            iou = calc_iou(pre_box,pb)
            if iou > merge_iou:
                b_iou.append(pb)
        if len(b_iou) == 0:
            return pre_box, time.time()-start_time, iter_cnt
            
        b_iou.append(pre_box)
        b_iou = np.array(b_iou)
        b_iou = b_iou[:, :8].reshape((-1, 2))
        rect = cv2.minAreaRect(b_iou)
        points = cv2.boxPoints(rect)
        points = points.reshape((-1))
        current_box = np.insert(points,len(points),score)
        #point_boxes.append(current_box)
        
        #####stop iou
        iou = calc_iou(pre_box, current_box)
        
        
        pre_box = current_box
        if show_log:
            print("iter_cnt",iter_cnt,iou)
        
        
    #point_boxes = np.array(point_boxes)
    #point_boxes = lanms.merge_quadrangle_n9(point_boxes.astype('float32'), nms_thres)
    IRB2box_time = time.time()-start_time
    if show_log:
        print("IRB2box time:", IRB2box_time)
    return current_box, IRB2box_time, iter_cnt

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    scores = dets[:, 8]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        anchor = (np.array(dets[i, 0:8]).reshape((4, 2))).astype(np.float32)
        size1 = cv2.contourArea(anchor)
        others = np.array(dets[order[1:],:])
        ovr = np.zeros((others.shape[0]))
        for j in range(others.shape[0]):
            proposal = (others[j,0:8].reshape((4,2))).astype(np.float32)
            size2 = cv2.contourArea(proposal)
            inter = cv2.intersectConvexConvex(anchor, proposal)
            inter_size = inter[0]
            iou = inter_size / (size1 + size2 - inter_size)
            ovr[j] = iou

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def detect(score_map, geo_map, timer, score_map_thresh=FLAGS.threshold, box_thresh=FLAGS.box_thresh, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        if FLAGS.IRB == False:
            return None, timer
        else:
            return None, timer, 0, 0

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    if FLAGS.IRB == False:
        return boxes, timer
    
    #########################IRB#########################
    #->:all box -> nms -> box 
    #1-> in box,4points
    #2->4box
    #3-> min area box
    point_boxes = []
    IRB2box_times = 0; iter_cnts = 0
    for b in boxes:
        poly = b[0:8].reshape((4, 2)).astype(np.int32)
        poly_h = max(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = max(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if max(poly_h,poly_w)>FLAGS.start_IRB_max_len:
            current_box, IRB2box_time, iter_cnt = IRB(b,score_map, geo_map,score_map_thresh,show_log=True,merge_iou=nms_thres)#,iter_max=10,iter_stop_iou=0.9)
        else:
            current_box = b
            IRB2box_time = 0; iter_cnt = 0
        point_boxes.append(current_box)
        IRB2box_times += IRB2box_time
        iter_cnts += iter_cnt
    point_boxes = np.array(point_boxes)
    if point_boxes.shape[0] != 0:
        point_boxes = point_boxes[point_boxes[:, 8] > box_thresh]
        point_boxes = point_boxes[py_cpu_nms(point_boxes.astype('float32'), nms_thres)]
    else:
        point_boxes = None
    return point_boxes, timer, IRB2box_times, iter_cnts


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    FLAGS.output_dir = FLAGS.output_dir + '_' + str(FLAGS.resize_ratio)
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        is_training = tf.placeholder(tf.bool, name='training_flag')
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry, _ = model.model(input_images, is_training=is_training, anchor_sizes=ANCHOR_SIZES, select_split_N=select_split_N,
                                                    is_select_background=FLAGS.is_select_background)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                model_path = FLAGS.checkpoint_path
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            IRB2box_times = 0
            iter_cnts = 0
            total_times = 0
            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im, re_ratio=FLAGS.resize_ratio)
                print(im.shape,im_resized.shape,(ratio_h, ratio_w) )

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized], is_training: False})
                timer['net'] = time.time() - start

                boxes, timer, IRB2box_time, iter_cnt = detect(score_map=score, geo_map=geometry, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                
                IRB2box_times += IRB2box_time
                total_times += duration
                iter_cnts += iter_cnt

                # save to file
                res_file = os.path.join(FLAGS.output_dir, 'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                with open(res_file, 'w') as f:
                    if boxes is not None:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
            print("----------------average time--------------")
            print("pre image:", total_times/len(im_fn_list))
            print("pre IRB time:", IRB2box_times/max(iter_cnts,1))
            print("pre IRB iter_cnt:", iter_cnts/len(im_fn_list))

if __name__ == '__main__':
    tf.app.run()
