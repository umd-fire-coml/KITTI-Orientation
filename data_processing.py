#! /usr/bin/env python3
import os
import math
import numpy as np
import copy
import tensorflow as tf
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize
from os.path import join
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from random import random
from orientation_converters import (angle_to_trisector_affinity, 
    alpha_to_multibin_orientation_confidence, angle_to_angle_normed, NUM_BIN)

# constants
NORM_H, NORM_W = 224, 224
TRAIN_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
KITTI_CLASSES = ['Cyclist', 'Tram', 'Person_sitting', 'Truck', 'Pedestrian', 'Van', 'Car', 'Misc', 'DontCare']
DIFFICULTY = ['easy', 'moderate', 'hard']
MIN_BBOX_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
NUMPY_TYPE = np.float32
VIEW_ANGLE_TOTAL_X = 1.4835298642
VIEW_ANGLE_TOTAL_Y = 0.55850536064

# store the average dimensions of different train classes
class_dims_means = {key: np.asarray([0, 0, 0]) for key in TRAIN_CLASSES}
# store the count of different train classes
class_counts = {key: 0 for key in TRAIN_CLASSES}

# this creates a list of dict for each obj from from the kitti train val directories
def get_all_objs_from_kitti_dir(label_dir, image_dir, difficulty='hard'):
    # get difficulty index
    DIFFICULTY_id = DIFFICULTY.index(difficulty)
    # store all objects
    all_objs = []

    for image_file in tqdm(os.listdir(image_dir)):
        label_file = image_file.replace('.png', '.txt')

        # each line represents an object
        for obj_line in open(join(label_dir, label_file)).readlines():
            obj_line_tokens = obj_line.strip().split(' ')
            class_name = obj_line_tokens[0]
            truncated = float(obj_line_tokens[1]) # Float from 0 (non-truncated) to 1 (truncated)
            occluded = int(obj_line_tokens[2]) # 0 = fully visible, 1 = partly occluded,
            ymin = int(float(obj_line_tokens[5]))
            ymax = int(float(obj_line_tokens[7]))
            bbox_height = ymax - ymin

            # filter objs based on TRAIN_CLASSES, MIN_HEIGHT, MAX_OCCLUSION, MAX_TRUNCATION
            if (class_name in TRAIN_CLASSES 
                and bbox_height > MIN_BBOX_HEIGHT[DIFFICULTY_id] 
                and occluded <= MAX_OCCLUSION[DIFFICULTY_id]
                and truncated <= MAX_TRUNCATION[DIFFICULTY_id]):

                obj = {
                        'image_file': image_file,
                        'class_name': class_name,
                        'class_id': KITTI_CLASSES.index(class_name),
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': float(obj_line_tokens[3]),
                        'xmin': int(float(obj_line_tokens[4])),
                        'ymin': ymin,
                        'xmax': int(float(obj_line_tokens[6])),
                        'ymax': ymax,
                        'height': float(obj_line_tokens[8]),
                        'width': float(obj_line_tokens[9]),
                        'length': float(obj_line_tokens[10]),
                        'dims': np.asarray([float(number) for number in obj_line_tokens[8:11]]),
                        'loc_x': float(obj_line_tokens[11]),
                        'loc_y': float(obj_line_tokens[12]),
                        'loc_z': float(obj_line_tokens[13]),
                        'rot_y': float(obj_line_tokens[14]),
                       }

                # Get multibin
                orientation, confidence = alpha_to_multibin_orientation_confidence(obj["alpha"])
                obj['multibin_orientation'] = orientation
                obj['multibin_confidence'] = confidence

                # Get tricosine
                obj['tricosine'] = angle_to_trisector_affinity(obj['alpha'])

                # Get camera view angle of the object
                center = ((obj['xmin'] + obj['xmax']) / 2, (obj['ymin'] + obj['ymax']) / 2)
                obj['view_angle'] = center[0] / NORM_W * VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X / 2)

                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                class_dims_means[obj['class_name']] = class_counts[obj['class_name']] * \
                                              class_dims_means[obj['class_name']] + obj['dims']
                class_counts[obj['class_name']] += 1
                # get the new average
                class_dims_means[obj['class_name']] /= class_counts[obj['class_name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data in obj dict format
    return all_objs

# get the bounding box,  values for the instance
# this automatically does flips
def prepare_generator_output(image_dir: str, obj, orientation: str):
    # Prepare image patch
    xmin = obj['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = obj['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = obj['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = obj['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

    # read image
    img = img_as_float(io.imread(join(image_dir, obj['image_file'])))
    img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1])
    # resize the image to standard size
    img = resize(img, (NORM_H, NORM_W), anti_aliasing=True)
    img = img.astype(NUMPY_TYPE)

    # Get the dimensions offset from average (basically zero centering the values)
    obj['dims'] = obj['dims'] - class_dims_means[obj['class_name']]
 
    # flip the image by random chance
    flip = random() < 0.5

    # flip image horizontally
    if flip:
        img = np.fliplr(img)
        if orientation == 'multibin':
            if 'multibin_orientation_flipped' not in obj:
                # Get orientation and confidence values for flip
                orientation_flipped, confidence_flipped = alpha_to_multibin_orientation_confidence(math.tau - obj["alpha"])
                obj['multibin_orientation_flipped'] = orientation_flipped
                obj['multibin_confidence_flipped'] = confidence_flipped
            return img, obj['multibin_orientation_flipped'], obj['multibin_confidence_flipped']

        elif orientation == 'tricosine':
            if 'tricosine_flipped' not in obj:
                obj['tricosine_flipped'] = angle_to_trisector_affinity(math.tau - obj['alpha'])
            return img, obj['tricosine_flipped']

        elif orientation == 'alpha':
            if 'alpha_normed_flipped' not in obj:
                obj['alpha_normed_flipped'] = angle_to_angle_normed(math.tau - obj['alpha'])
            return img, obj['alpha_normed_flipped']  

        elif orientation == 'rot_y':
            if 'rot_y_normed_flipped' not in obj:
                obj['rot_y_normed_flipped'] = angle_to_angle_normed(math.tau - obj['rot_y'])
            return img, obj['rot_y_normed_flipped']

        else:
            raise Exception("No such orientation type: %s" % orientation)
    else:
        if orientation == 'multibin':
            if 'multibin_orientation' not in obj:
                # Get orientation and confidence values for flip
                orientation, confidence = alpha_to_multibin_orientation_confidence(obj["alpha"])
                obj['multibin_orientation'] = orientation
                obj['multibin_confidence'] = confidence
            return img, obj['multibin_orientation'], obj['multibin_confidence']
            
        elif orientation == 'tricosine':
            if 'tricosine' not in obj:
                obj['tricosine'] = angle_to_trisector_affinity(obj['alpha'])
            return img, obj['tricosine']

        elif orientation == 'alpha':
            if 'alpha_normed' not in obj:
                obj['alpha_normed'] = angle_to_angle_normed(obj['alpha'])
            return img, obj['alpha_normed'] 

        elif orientation == 'rot_y':
            if 'rot_y_normed' not in obj:
                obj['rot_y_normed'] = angle_to_angle_normed(obj['rot_y'])
            return img, obj['rot_y_normed']

        else:
            raise Exception("No such orientation type: %s" % orientation)

class KittiGenerator(Sequence):
    '''Creates A KittiGenerator Sequence
    Args:
        label_dir (str) : path to the directory with labels
        image_dir (str) : path to the image directory
        mode (str): tells whether to be in train, val or all mode
        batch_size (int) : tells batchsize to use
        orientation_type (str): type of oridentation multibin, tricosine, alpha, or rot_y
        val_split (float): what percentage data reserved for validation
    '''

    def __init__(self, label_dir: str = 'dataset/training/label_2/',
                 image_dir: str = 'dataset/training/image_2/',
                 mode: str = "train",
                 batch_size: int = 8,
                 orientation_type: str = "multibin",
                 val_split: float = 0.0):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.all_objs = get_all_objs_from_kitti_dir(label_dir, image_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.orientation_type = orientation_type
        self.num_objs = len(self.all_objs)  # number of objects
        self.obj_ids = list(range(self.num_objs))  # list of all object indexes for the generator
        if val_split > 0.0:
            assert mode != 'all' and val_split < 1.0
            cutoff = int(val_split * self.num_objs)
            if self.mode == "train":
                self.obj_ids = self.obj_ids[cutoff:]
                self.num_objs = len(self.obj_ids)
            elif self.mode == "val":
                self.obj_ids = self.obj_ids[:cutoff]
                self.num_objs = len(self.obj_ids)
            else:
                assert False, "invalid mode"
        self.on_epoch_end()

    def __len__(self):
        return self.num_objs // self.batch_size

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size  # start of key index
        r_bound = l_bound + self.batch_size  # end of key index
        r_bound = r_bound if r_bound < self.num_objs else self.num_objs  # check for key index overflow
        num_batch_objs = r_bound - l_bound
        img_batch = np.empty((num_batch_objs, NORM_H, NORM_W, 3))  # batch of images

        if self.orientation_type == "multibin":
            # prepare output tensors
            orientation_batch = np.empty((num_batch_objs, NUM_BIN, 2))
            confidence_batch = np.empty((num_batch_objs, NUM_BIN))
            
            for i, key in enumerate(self.obj_ids[l_bound:r_bound]):
                image, orientation, confidence = prepare_generator_output(self.image_dir, self.all_objs[key], self.orientation_type)
                img_batch[i] = image
                orientation_batch[i] = orientation
                confidence_batch[i] = confidence

            return img_batch, orientation_batch.astype(NUMPY_TYPE), confidence_batch.astype(NUMPY_TYPE)

        elif self.orientation_type == 'tricosine':

            tricosine_batch = np.empty((num_batch_objs, 3))

            for i, key in enumerate(self.obj_ids[l_bound:r_bound]):
                image, tricosine = prepare_generator_output(self.image_dir, self.all_objs[key], self.orientation_type)
                img_batch[i] = image
                tricosine_batch[i] = tricosine

            return img_batch, tricosine_batch.astype(NUMPY_TYPE)

        elif self.orientation_type == "alpha" or self.orientation_type == 'rot_y':

            angle_batch = np.empty((num_batch_objs, 1))

            for i, key in enumerate(self.obj_ids[l_bound:r_bound]):
                image, angle = prepare_generator_output(self.image_dir, self.all_objs[key], self.orientation_type)
                img_batch[i] = image
                angle_batch[i] = angle

            return img_batch, angle_batch.astype(NUMPY_TYPE)
        else:
            raise Exception("Invalid Orientation Type")

    def on_epoch_end(self):
        np.random.shuffle(self.obj_ids)

    def __str__(self):
        return "KittiDatagenerator:<size %d, orientation_type: %s, image_dir:%s, label_dir:%s, epoch:%d>" % (
        len(self), self.orientation_type, self.image_dir, self.label_dir, self.epochs)