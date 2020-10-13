#! python3
import os
import math
import numpy as np
import cv2
import copy
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import warnings
#import tensorflow as tf
#####
# Training setting
BIN, OVERLAP = 2, 0.1
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']
NUM_CATS = 4

# make sure that math.tau isn't causint issues
def alpha_rad_to_tricoine(alpha_rad, sectors=3):
    SECTOR_WIDTH = math.tau / sectors
    sector_affinity = np.full((sectors,), -1.0)
    # calculate the center sector affinity
    center_sector_id = alpha_rad // SECTOR_WIDTH
    center_sector_offset = (SECTOR_WIDTH / 2) - (alpha_rad % SECTOR_WIDTH)
    center_sector_affinity = math.cos(center_sector_offset)
    sector_affinity[center_sector_id] = center_sector_affinity
    # calculate the left sector affinity
    l_sector_id = (center_sector_id - 1) % sectors  # if -1 then we get 2
    l_sector_offset = (alpha_rad % SECTOR_WIDTH) + (SECTOR_WIDTH / 2)
    l_sector_affinity = math.cos(l_sector_offset)
    sector_affinity[l_sector_id] = l_sector_affinity
    # calculate the left sector affinity
    r_sector_id = (center_sector_id + 1) % sectors
    r_sector_offset = (SECTOR_WIDTH - (alpha_rad %
                                       SECTOR_WIDTH)) + (SECTOR_WIDTH / 2)
    r_sector_affinity = math.cos(r_sector_offset)
    sector_affinity[r_sector_id] = r_sector_affinity
    return sector_affinity


def tricosine_to_alpha_rad(sector_affinity,sectors=3):
    # calculate center sector offset
    SECTOR_WIDTH = math.tau / sectors
    center_sector_id = np.argmax(sector_affinity)
    center_sector_affinity = sector_affinity[center_sector_id]
    center_sector_offset = math.acos(center_sector_affinity)

    # calculate left sector offset
    l_sector_id = (center_sector_id - 1) % sectors
    l_sector_affinity = sector_affinity[l_sector_id]
    l_sector_offset = math.acos(l_sector_affinity)

    # calculate right sector offset
    r_sector_id = (center_sector_id + 1) % sectors
    r_sector_affinity = sector_affinity[r_sector_id]
    r_sector_offset = math.acos(r_sector_affinity)

    # refine down to the angle
    alpha_rad_from_center_sector = center_sector_id * \
        SECTOR_WIDTH + (SECTOR_WIDTH/2)  # middle of center
    # calculuate angle from center sector (based on direction signal)
    if l_sector_offset < r_sector_offset:
        alpha_rad_from_center_sector = (
            alpha_rad_from_center_sector - center_sector_offset) % math.tau
    elif l_sector_offset > r_sector_offset:
        alpha_rad_from_center_sector = (
            alpha_rad_from_center_sector + center_sector_offset) % math.tau
    # calculuate angle from left sector
    alpha_rad_from_l_sector = (
        l_sector_id * SECTOR_WIDTH + (SECTOR_WIDTH/2) + l_sector_offset) % math.tau
    # calculuate angle from right sector
    alpha_rad_from_r_sector = (
        r_sector_id * SECTOR_WIDTH + (SECTOR_WIDTH/2) - r_sector_offset) % math.tau
    alpha_rads = [alpha_rad_from_center_sector,
                  alpha_rad_from_l_sector, alpha_rad_from_r_sector]
    # calculuate the mean angle
    sum_sin_alpha_rads = np.sum(np.sin(alpha_rads))
    sum_cos_alpha_rads = np.sum(np.cos(alpha_rads))
    mean_alpha_rads = np.arctan2(sum_sin_alpha_rads, sum_cos_alpha_rads)
    return mean_alpha_rads

def angle2cat(angle:int, n:int = 4)->float:
    if angle<0:
        angle += np.tau
    return int(angle/(np.tau/n))

def compute_anchors(angle):
    # angle is the new_alpha angle between 0 and 2pi
    anchors = []
    wedge = 2.*np.pi/BIN  # angle size of each bin, i.e. 180 deg
    # round down with int, tells me which bin the angle belongs to, gets to be either 0 or 1
    l_index = int(angle/wedge)
    r_index = l_index + 1  # get to be either 1 or 2

    # (angle - l_index*wedge) is the +offset angle from start of the wedge, l_index*wedge is either 0 or 180
    # wedge/2 * (1+OVERLAP/2) is 90 deg * 1.05
    # basically check if the angle is within majority part of the current wedge
    if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
        # append the bin index of the angle, and the +offset angle from start of the wedge
        anchors.append([l_index, angle - l_index*wedge])

    # r_index*wedge - angle is the -offset angle from start of the next wedge, r_index*wedge is either 180 or 360 deg
    # wedge/2 * (1+OVERLAP/2) is 90 deg * 1.05
    # basically check if the angle is also within majority part of the next wedge
    if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
        anchors.append([r_index % BIN, angle - r_index*wedge])

    return anchors
# this creates the full dict from the train val directories
def parse_annotation(label_dir, image_dir,mode = 'train'):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in sorted(os.listdir(label_dir)):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            # if the class is in VEHICLES and not truncated not occluded
            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                # offset to make new_alpha, so that if car is head facing the camera, new_alpha = pi
                # , and if car is back facing the camera, new_alpha = 0
                new_alpha = float(line[3]) + np.pi/2.
                # make new_alpha always >= 0
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                # make new_alpha always <= 2pi, equivalent to if new_alpha > 2.*np.pi: new_alpha = new_alpha - 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name': line[0],  # class
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                       }

                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                dims_avg[obj['name']] = dims_cnt[obj['name']] * \
                    dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                # get the new average
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data in obj dict format

    # flip data
    for obj in all_objs:

        # Get the dimensions offset from average (basically zero centering the values)
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Get orientation and confidence values for no flip
        # set all values as zeros for each orientation (2x2 values) and conf  (2 values, each value represents the sector)
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        # get the sector id and offset from center of each sector if its within +-94.5 deg from the center
        anchors = compute_anchors(obj['new_alpha'])

        for anchor in anchors:
            # compute the cos and sin of the offset angles
            orientation[anchor[0]] = np.array(
                [np.cos(anchor[1]), np.sin(anchor[1])])
            # set confidence of the sector to 1
            confidence[anchor[0]] = 1.

        # if in both sectors, then each confidence is 1/2, this makes sure sum of confidence adds up to 1
        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # add our implementation here
        obj['tri_sector_affinity'] =  alpha_rad_to_tricoine(
            obj['new_alpha'])
        obj['alpha_cat'] = angle2cat(obj['new_alpha'])
        # Get orientation and confidence values for flip
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        # flip the camera angle across 0 deg
        anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
        for anchor in anchors:
            orientation[anchor[0]] = np.array(
                [np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence
        # add our implementation here
        obj['tri_sector_affinity_flipped'] = alpha_rad_to_tricoine(
            2.*np.pi - obj - obj['new_alpha'])

    return all_objs

# get the bounding box,  values for the instance
# this automatically does flips
def prepare_input_and_output(image_dir, train_inst):
    # Prepare image patch
    xmin = train_inst['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    img = cv2.imread(image_dir + train_inst['image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # crop the image using the obj bounding box, deepcopy to prevent memory sharing
    img = copy.deepcopy(img[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)

    # flip the image by random chance
    flip = np.random.binomial(1, .5)
    # flip image horizonatally
    if flip > 0.5:
        img = cv2.flip(img, 1)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    # zero center the image values around these (avg?) RGB values
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    # convert to rgb
    img = img[:,:,::-1]

    # if the image crop is flipped also flip the orientation values
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


class KittiGenerator(Sequence):

    '''Creates A KittiGenerator Sequence
    Args:
        label_dir (str) : path to the directory with labels
        image_dir (str) : path to the image directory
        mode (str): tells whether to be in train or test mode
        batch_size (int) : tells batchsize to use
    '''
    def __init__(self,label_dir:str,image_dir:str,mode = "train",batch_size = 8,**kwargs):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.all_objs = parse_annotation(label_dir,image_dir,mode)
        self.mode = mode
        self.batch_size = batch_size
        if mode!='test':
            warnings.warn("testing mode has not been inplemented yet")
        if mode!='val':
            warnings.warn("validation mode has not been inplemented yet")
        self._clen = len(self)
        self._keys = range(self._clen)
        np.random.shuffle(self._keys)
        self.alpha_m = False
        self.epochs = 0
        self._idx = 0
        if 'alpha' in kwargs and kwargs['alpha']:
            warnings.warn("alpha mode has not been inplemented yet")
            self.alpha_m = True

    def __len__(self)->int:
        return len(self.all_objs)

    def __getitem__(self,idx):
        l_bound = idx
        r_bound = self.batch_size+idx 
        r_bound = r_bound if r_bound<self._clen else self._clen
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))  # batch of images
        d_batch = np.zeros((r_bound - l_bound, 3))  # batch of dimensions
        # batch of cos,sin values for each bin
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        # batch of confs for each bin
        c_batch = np.zeros((r_bound - l_bound, BIN))
        acat_batch = np.zeros((r_bound-l_bound,NUM_CATS))
        currt_inst = 0
        for key in self._keys[l_bound:r_bound]:
            image, dimension, orientation, confidence = prepare_input_and_output(
                self.image_dir, self.all_objs[key])
            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence
            if self.alpha_m:
                acat_batch[currt_inst,angle2cat(self.all_objs[key]['new_alpha'])] = 1
            currt_inst += 1
        if self.alpha_m:
            raise Exception("ALPHA MODE UNIMPLEMENTED")
            #return x_batch, [d_batch, o_batch, c_batch,acat_batch]
        return x_batch, [d_batch, o_batch, c_batch]

    def on_epoch_end(self):
        print("initializing next epoch")
        np.random.shuffle(self._keys)
        self.epochs+=1
        self._idx = 0
    
    def __str__(self):
        return "KittiDatagenerator:<size %d,image_dir:%s,label_dir:%s,epoch:%d>"%(len(self),self.image_dir,self.label_dir,self.epochs)
    
    def __next__(self):
        result = self.__getitem__(self._idx)
        self._idx += len(result)
        if self._idx>=len(self):
            self.on_epoch_end()
        return result
    def get_tf_handle(self)->tf.data.Dataset:
        if self.alpha_m:
            raise Exception("alpha mode has not been implemented")
        output_shape = {tf.TensorShape([(NORM_H,NORM_W)]),tf.TensorShape([(3,),(BIN,2),(BIN,)])} 
        return tf.data.Dataset.from_generator(generator=self,output_types=(tf.float32),output_shapes=output_shape)
