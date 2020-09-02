#!/usr/bin/env python
import os
import rospy

data = 'real_test'
use_regression = False
use_delta = False
num_eval = -1
concur = None

gpu = '0'
mode = 'detect'
subfile = "inputs"

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
print('Using GPU {}.'.format(gpu))

import roslib
import rospy
import rospkg

import sys
import datetime
import glob
import time
import numpy as np
from config import Config
import utils
import model as modellib
from get_data import MYDataset
import _pickle as cPickle
from train import ScenesConfig
from pose_thread import generate_pose
from pose_thread import Concur
from std_srvs.srv import SetBool
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import threading
import time

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

srv_activated = True


class InferenceConfig(ScenesConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    COORD_USE_REGRESSION = use_regression
    if COORD_USE_REGRESSION:
        COORD_REGRESS_LOSS = 'Soft_L1'
    else:
        COORD_NUM_BINS = 32
    COORD_USE_DELTA = use_delta

    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False


def RT_filter(pred_RTs, detect_range):
    RT_num = len(pred_RTs)
    filtered_RTs = []
    for i in range(RT_num):
        RT = pred_RTs[i]
        T1 = RT[:3, 3]
        obj_norm = np.linalg.norm(T1) * 100  # in cm
        if (obj_norm > detect_range):
            continue
        else:
            filtered_RTs.append(RT)
    return filtered_RTs


def position_diff(RT1, RT2):
    T1 = RT1[:3, 3]
    T2 = RT2[:3, 3]
    diff = np.linalg.norm(T1 - T2)
    return diff


class NOCS_Estimation:
    def __init__(self):
        self.config = InferenceConfig()
        self.config.display()

        r = rospkg.RosPack()
        self.pack_path = r.get_path('nocs_srv')
        #  real classes
        self.coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light',
                           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                           'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                           'kite', 'baseball bat', 'baseball glove', 'skateboard',
                           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                           'teddy bear', 'hair drier', 'toothbrush']

        self.synset_names = ['BG',  # 0
                             'bottle',  # 1
                             'bowl',  # 2
                             'camera',  # 3
                             'can',  # 4
                             'laptop',  # 5
                             'mug'  # 6
                             ]

        self.class_map = {
            'bottle': 'bottle',
            'bowl': 'bowl',
            'cup': 'mug',
            'laptop': 'laptop',
        }

        coco_cls_ids = []
        for coco_cls in self.class_map:
            ind = self.coco_names.index(coco_cls)
            coco_cls_ids.append(ind)
        self.config.display()

        tf_config = tf.ConfigProto()
        self.sess = tf.Session(config=tf_config)
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        # Recreate the model in inference mode
        self.model = modellib.MaskRCNN(mode="inference",
                                       config=self.config,
                                       model_dir=MODEL_DIR)

        self.ckpt_path = rospy.get_param('~ckpt_path', "logs/nocs_rcnn_res50_bin32.h5")

        # Load trained weights (fill in path to trained weights here)
        model_path = r.get_path('nocs_srv') + '/' + self.ckpt_path
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        self.model.load_weights(model_path, by_name=True)

        now = datetime.datetime.now()
        self.save_dir = os.path.join(r.get_path('nocs_srv'), 'output')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if rospy.has_param('~intrinsics_fx') and rospy.has_param('~intrinsics_fy') and rospy.has_param(
                '~intrinsics_x0') and rospy.has_param('~intrinsics_y0'):
            self.intrinsics_fx = rospy.get_param('~intrinsics_fx')
            self.intrinsics_fy = rospy.get_param('~intrinsics_fy')
            self.intrinsics_x0 = rospy.get_param('~intrinsics_x0')
            self.intrinsics_y0 = rospy.get_param('~intrinsics_y0')
        else:
            rospy.logerr("Parameters of intrinsics not set.")

        # fetch robot:
        # self.intrinsics = np.array([[538.9424, 0, 327.98415], [0., 537.60248,  238.83914], [0., 0., 1.]])    #fetch calibrated
        # intrinsics = np.array([[554.257, 0.0, 320.5], [0.0, 554.2547, 240.5], [0., 0., 1.]])  # fetch sim
        self.intrinsics = np.array(
            [[self.intrinsics_fx, 0.0, self.intrinsics_x0], [0.0, self.intrinsics_fy, self.intrinsics_y0],
             [0., 0., 1.]])

        self.draw = rospy.get_param('~draw', True)
        self.detect_range = rospy.get_param('~detect_range', 150)
        if rospy.has_param('~camera_optical_frame'):
            self.camera_optical_frame = rospy.get_param('~camera_optical_frame', "head_camera_rgb_optical_frame")
        else:
            rospy.logerr("Parameters \'camera_optical_frame\' not set.")

        self.concur = Concur()
        self.first_call = True
        s = rospy.Service('estimate_pose_nocs', SetBool, self.call_srv)

        rospy.loginfo("NOCS Estimation initialized.")

    def start_estimation(self):
        n = 3
        log_file = os.path.join(self.save_dir, 'error_log.txt')
        self.f_log = open(log_file, 'w')
        image_id = 0
        last_estimation = None
        while n > 0:
            n = n - 1
            dataset_my = MYDataset(self.synset_names, 'val', self.config)

            data_path = self.pack_path + '/' + subfile
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            load_data_res = dataset_my.load_data(data_path)
            if not load_data_res:
                rospy.logerr("Load data Failed. Check path of the input images.")
                return False
            dataset_my.prepare(self.class_map)
            dataset = dataset_my

            print('*' * 50)
            image_start = time.time()
            image_path = dataset.image_info[image_id]["path"]

            result = {}
            image = dataset.load_image(image_id)
            depth = dataset.load_depth(image_id)

            result['image_id'] = image_id
            result['image_path'] = image_path

            image_path_parsing = image_path.split('/')
            ## detection
            start = time.time()
            with self.graph.as_default():
                set_session(self.sess)
                detect_result = self.model.detect([image], verbose=0)
            r = detect_result[0]
            elapsed = time.time() - start

            print('\nDetection takes {:03f}s.'.format(elapsed))
            result['pred_class_ids'] = r['class_ids']
            result['pred_bboxes'] = r['rois']
            result['pred_RTs'] = None
            result['pred_scores'] = r['scores']

            if len(r['class_ids']) == 0:
                rospy.logwarn('Nothing detected.')
                dataset_my.detect_finished()
                continue

            print('Aligning predictions...')
            start = time.time()
            result['pred_RTs'], result['pred_scales'], error_message, elapses = utils.align(r['class_ids'],
                                                                                            r['masks'],
                                                                                            r['coords'],
                                                                                            depth,
                                                                                            self.intrinsics,
                                                                                            self.synset_names,
                                                                                            image_path)
            print('New alignment takes {:03f}s.'.format(time.time() - start))
            print("The predicted transfrom matrix:")
            print(result['pred_RTs'])
            filtered_RTs = RT_filter(result['pred_RTs'], self.detect_range)
            if len(filtered_RTs) == 0:
                rospy.logwarn('No detected object in %d cm.', self.detect_range)
                dataset_my.detect_finished()
                continue

            if last_estimation is not None:
                pose_diff = position_diff(last_estimation, filtered_RTs[0])
                if pose_diff < 0.05:
                    n = 0
                    self.concur.set_T(filtered_RTs[0], self.camera_optical_frame)
                    if self.first_call:
                        self.concur.start()
                        self.first_call = False
                    else:
                        self.concur.resume()
                else:
                    rospy.loginfo('The pose estimation is not stable. Distance to last estimation: %f m.', pose_diff)
            last_estimation = filtered_RTs[0]
            if len(error_message):
                self.f_log.write(error_message)

            if self.draw:
                draw_rgb = False
                utils.draw_detections_wogt(image, self.save_dir, data,
                                           image_path_parsing[-2] + '_' + image_path_parsing[-1], self.intrinsics,
                                           self.synset_names, draw_rgb,
                                           r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'],
                                           r['scores'], result['pred_scales'])

            path_parse = image_path.split('/')
            image_short_path = '_'.join(path_parse[-3:])

            save_path = os.path.join(self.save_dir, 'results_{}.pkl'.format(image_short_path))
            with open(save_path, 'wb') as f:
                cPickle.dump(result, f)
            print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))

            elapsed = time.time() - image_start
            print('Takes {} to finish this image.'.format(elapsed))
            print('Alignment time: ', elapses)
            print('\n')

            dataset_my.detect_finished()

        self.f_log.close()
        return True

    def call_srv(self, req):
        if req.data is False:
            self.concur.pause()
            return [True, "Finished 6D pose estimation service"]
        elif req.data is True:
            res = self.start_estimation()
            return [res, "Started 6D pose estimation service"]


if __name__ == '__main__':
    rospy.init_node('estimate_pose_nocs_server')
    est = NOCS_Estimation()
    rospy.spin()
