#!/usr/bin/env python  
import roslib
import rospy
import cv2
import message_filters
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from distutils.version import LooseVersion
import pandas as pd
import numpy as np

import os
import sys
import datetime

from skimage import exposure
import time
import glob
import numpy as np
import cv2
import glob
from config import Config
import utils

sys.path.append('./cocoapi/PythonAPI')
from pycocotools.coco import COCO

input_flag = False
width = 640
height = 480
dsize = (width, height)


class MYDataset(utils.Dataset):
    """Generates the NOCS dataset.
    """

    def __init__(self, synset_names, subset, config=Config()):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        self.subset = subset

        self.config = config

        self.source_image_ids = {}

        # Add classes
        for i, obj_name in enumerate(synset_names):
            if i == 0:  ## class 0 is bg class
                continue
            self.add_class("BG", i, obj_name)  ## class id starts with 1

    def callback(self, imaged, imagec):
        global input_flag
        if input_flag:
            return
        try:
            NewImg = None
            if imaged.encoding == "32FC1":
                cv_image = self.CvImg.imgmsg_to_cv2(imaged, "32FC1")
                print("Depth image encoding: 32FC1")
                image = np.array(cv_image, dtype=np.float)
                image = image * 1000  # unit: m to mm
                NewImg = np.round(image).astype(np.uint16)
            elif imaged.encoding == "16UC1":
                NewImg = self.CvImg.imgmsg_to_cv2(imaged, "16UC1")
                print("Depth image encoding: 16UC1")
            else:
                rospy.logerr("Depth image encoding is neither 32FC1 nor 16UC1")
                return
            img_name = "_depth.png"
            img_path = os.path.join(self.image_dir, img_name)
            cv2.imwrite(img_path, NewImg)
            print("saved ", img_name)
        except CvBridgeError as e:
            print(e)
        try:
            NewImgd = self.CvImg.imgmsg_to_cv2(imagec, "bgr8")
            img_name = "_color.png"
            img_path = os.path.join(self.image_dir, img_name)
            cv2.imwrite(img_path, NewImgd)
            print("saved ", img_name)
        except CvBridgeError as e1:
            print(e1)
        input_flag = True

    def load_data(self, dataset_dir):
        self.image_dir = dataset_dir

        if rospy.has_param('~rgb_topic') and rospy.has_param('~dep_topic'):
            dep_topic = rospy.get_param('~dep_topic')
            rgb_topic = rospy.get_param('~rgb_topic')
        else:
            rospy.logerr("RGBD camera topics not set.")
        self.CvImg = CvBridge()
        skDep = message_filters.Subscriber(dep_topic, Image)
        skImg = message_filters.Subscriber(rgb_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer([skDep, skImg], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

        i = 0
        while not input_flag:
            time.sleep(2.5)
            i = i + 1
            print("Waiting rgbd topics ... ", i)
            if i > 10:
                print("no dep/rgb image subcribed for 2.5*10 seconds: exit the detection")
                return False
        self.load_my_scenes(dataset_dir)
        return True

    def detect_finished(self):
        global input_flag
        input_flag = False

    def load_my_scenes(self, dataset_dir):
        """Load a subset of the dataset.
        dataset_dir: The root directory of the  dataset.
        subset: What to load (train, val, test)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """
        # global i

        source = "My"

        self.image_dir = dataset_dir + '/'
        color_path = self.image_dir + '_color.png'
        depth_path = self.image_dir + '_depth.png'

        width = self.config.IMAGE_MAX_DIM
        height = self.config.IMAGE_MIN_DIM

        self.add_image(
            source=source,
            image_id=0,
            path=self.image_dir,
            width=width,
            height=height)

        return True

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            image_path = info["path"] + '_color.png'
            assert os.path.exists(image_path), "{} is missing".format(image_path)

            # depth_path = info["path"] + '_depth.png'
        elif info["source"] == 'coco':
            image_path = info["path"]
        elif info["source"] == "My":
            image_path = info["path"] + '_color.png'
        else:
            assert False, "[ Error ]: Unknown image source: {}".format(info["source"])

        print("image_path: ", image_path)
        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def load_depth(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real", "My"]:
            depth_path = info["path"] + '_depth.png'
            depth = cv2.imread(depth_path, -1)

            if len(depth.shape) == 3:
                # This is encoded depth image, let's convert
                depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(
                    depth[:, :, 2])  # NOTE: RGB is actually BGR in opencv
                depth16 = depth16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == 'uint16':
                depth16 = depth
            else:
                assert False, '[ Error ]: Unsupported depth type.'
        else:
            depth16 = None

        return depth16

    def image_reference(self, image_id):
        """Return the object data of the image."""
        info = self.image_info[image_id]
        if info["source"] in ["ShapeNetTOI", "Real"]:
            return info["inst_dict"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_objs(self, image_id, is_normalized):
        info = self.image_info[image_id]
        meta_path = info["path"] + '_meta.txt'
        inst_dict = info["inst_dict"]

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        Vs = []
        Fs = []
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            inst_id = int(words[0])
            if not inst_id in inst_dict:
                continue

            if len(words) == 3:  ## real data
                if words[2][-3:] == 'npz':
                    obj_name = words[2].replace('.npz', '_norm.obj')
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', obj_name)
                else:
                    mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_' + self.subset, words[2] + '.obj')
                flip_flag = False
            else:
                assert len(words) == 4  ## synthetic data
                mesh_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'model.obj')
                flip_flag = True

            vertices, faces = utils.load_mesh(mesh_file, is_normalized, flip_flag)
            Vs.append(vertices)
            Fs.append(faces)

        return Vs, Fs

    def process_data(self, mask_im, coord_map, inst_dict, meta_path, load_RT=False):
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata == 255] = -1
        assert (np.unique(cdata).shape[0] < 20)

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        with open(meta_path, 'r') as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')

            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == 'npz':
                    npz_path = os.path.join(self.config.OBJ_MODEL_DIR, 'real_val', words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file['scale']
                else:
                    bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, 'real_' + self.subset, words[2] + '.txt')
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, self.subset, words[2], words[3], 'bbox.txt')
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]

        i = 0

        # delete ids of background objects and non-existing objects 
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]

        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            i += 1

        # print('before: ', inst_dict)

        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        class_ids = class_ids[:i]
        scales = scales[:i]

        return masks, coords, class_ids, scales

    def load_mask(self, image_id):
        """Generate instance masks for the objects in the image with the given ID.
        """
        info = self.image_info[image_id]
        # masks, coords, class_ids, scales, domain_label = None, None, None, None, None

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0  ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'

            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)

            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, (2, 1, 0)]

            masks, coords, class_ids, scales = self.process_data(mask_im, coord_map, inst_dict, meta_path)


        elif info["source"] == "coco":
            domain_label = 1  ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                        info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            if class_ids:
                masks = np.stack(instance_masks, axis=2)
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)

            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape + (3,), dtype=np.float32)
            scales = np.ones((len(class_ids), 3), dtype=np.float32)
            # print('\nwithout augmented, masks shape: {}'.format(masks.shape))
        else:
            assert False

        return masks, coords, class_ids, scales, domain_label

    def load_augment_data(self, image_id):
        """Generate augmented data for the image with the given ID.
        """
        info = self.image_info[image_id]
        image = self.load_image(image_id)

        # apply random gamma correction to the image
        gamma = np.random.uniform(0.8, 1)
        gain = np.random.uniform(0.8, 1)
        image = exposure.adjust_gamma(image, gamma, gain)

        # generate random rotation degree
        rotate_degree = np.random.uniform(-5, 5)

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0  ## has coordinate map loss

            mask_path = info["path"] + '_mask.png'
            coord_path = info["path"] + '_coord.png'
            inst_dict = info['inst_dict']
            meta_path = info["path"] + '_meta.txt'

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, ::-1]

            image, mask_im, coord_map = utils.rotate_and_crop_images(image,
                                                                     masks=mask_im,
                                                                     coords=coord_map,
                                                                     rotate_degree=rotate_degree)
            masks, coords, class_ids, scales = self.process_data(mask_im, coord_map, inst_dict, meta_path)
        elif info["source"] == "coco":
            domain_label = 1  ## no coordinate map loss

            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = utils.annToMask(annotation, info["height"],
                                        info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            masks = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)

            # print('\nbefore augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))
            image, masks = utils.rotate_and_crop_images(image,
                                                        masks=masks,
                                                        coords=None,
                                                        rotate_degree=rotate_degree)

            # print('\nafter augmented, image shape: {}, masks shape: {}'.format(image.shape, masks.shape))

            if len(masks.shape) == 2:
                masks = masks[:, :, np.newaxis]

            final_masks = []
            final_class_ids = []
            for i in range(masks.shape[-1]):
                m = masks[:, :, i]
                if m.max() < 1:
                    continue
                final_masks.append(m)
                final_class_ids.append(class_ids[i])

            if final_class_ids:
                masks = np.stack(final_masks, axis=2)
                class_ids = np.array(final_class_ids, dtype=np.int32)
            else:
                # Call super class to return an empty mask
                masks = np.empty([0, 0, 0])
                class_ids = np.empty([0], np.int32)

            # use zero arrays as coord map for COCO images
            coords = np.zeros(masks.shape + (3,), dtype=np.float32)
            scales = np.ones((len(class_ids), 3), dtype=np.float32)

        else:
            assert False

        return image, masks, coords, class_ids, scales, domain_label
