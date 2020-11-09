#! /bin/python3
import os
import math
import numpy as np
import cv2
import copy
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import warnings
from pathlib2 import Path
from datetime import datetime
from tqdm import tqdm
BIN, OVERLAP = 2, 0.1
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist']
ALL_OBJ =  ['Cyclist','Tram','Person_sitting','Truck','Pedestrian','Van','Car','Misc','DontCare']
NUM_CATS = 4
NUMPY_TYPE = np.float32
VIEW_ANGLE_TOTAL_X = 1.4835298642
VIEW_ANGLE_TOTAL_Y = 0.55850536064

# make sure that math.tau isn't causint issues
def alpha_rad_to_tricoine(alpha_rad, sectors=3):
    SECTOR_WIDTH = math.tau / sectors
    sector_affinity = np.full((sectors,), -1.0)
    # calculate the center sector affinity
    center_sector_id = alpha_rad // SECTOR_WIDTH
    center_sector_offset = (SECTOR_WIDTH / 2) - (alpha_rad % SECTOR_WIDTH)
    center_sector_affinity = math.cos(center_sector_offset)
    sector_affinity[int(center_sector_id)] = center_sector_affinity
    # calculate the left sector affinity
    l_sector_id = (center_sector_id - 1) % sectors  # if -1 then we get 2
    l_sector_offset = (alpha_rad % SECTOR_WIDTH) + (SECTOR_WIDTH / 2)
    l_sector_affinity = math.cos(l_sector_offset)
    sector_affinity[int(l_sector_id)] = l_sector_affinity
    # calculate the left sector affinity
    r_sector_id = (center_sector_id + 1) % sectors
    r_sector_offset = (SECTOR_WIDTH - (alpha_rad %
                                       SECTOR_WIDTH)) + (SECTOR_WIDTH / 2)
    r_sector_affinity = math.cos(r_sector_offset)
    sector_affinity[int(r_sector_id)] = r_sector_affinity
    return sector_affinity


def tricosine_to_alpha_rad(sector_affinity, sectors=3):
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

def angle2cat(angle:int, n:int = 4):
    if angle<0:
        angle += math.tau
    idx = int(angle/(math.tau/n))
    arr = np.zeros(n).astype(NUMPY_TYPE)
    arr[idx] = 1.0
    return arr

def qualityaware(distr_cats,ry_cats:int=4):
    #Spread out the value for quality-aware loss
    section_number = np.where(np.isclose(distr_cats,1.0))[0]
    cat_num = (int(ry_cats - 2)/2) # Remove the 1 and 0 sector, and divide by 2
    #
    left = section_number - 1
    right = section_number + 1
    if right == ry_cats:
        right = 0
    if left == -1:
        left = ry_cats-1
            
    for i in range(int(cat_num)):
        distr_cats[right] = 1 - 1/(cat_num + 1) * (i+1)
        distr_cats[left] = 1 - 1/(cat_num + 1) * (i+1)
        right += 1
        left -= 1
        if right == ry_cats:
            right = 0
        if left == -1:
            left = ry_cats-1
    return distr_cats

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
def parse_annotation(label_dir, image_dir,mode = 'train',num_alpha_sectors=4,num_rot_y_sectors=4):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in tqdm(sorted(os.listdir(label_dir))):
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

                obj = {'class_name': line[0],  # class
                        'class_id' : ALL_OBJ.index(line[0]),
                       'image_path': Path(image_file),
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha,
                       'alpha':float(line[3]),
                       'rot_y': float(line[14])
                       }

                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                dims_avg[obj['class_name']] = dims_cnt[obj['class_name']] * \
                    dims_avg[obj['class_name']] + obj['dims']
                dims_cnt[obj['class_name']] += 1
                # get the new average
                dims_avg[obj['class_name']] /= dims_cnt[obj['class_name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data in obj dict format

    # flip data
    for obj in all_objs:

        # Get the dimensions offset from average (basically zero centering the values)
        obj['dims'] = obj['dims'] - dims_avg[obj['class_name']]

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

        obj['multibin_orientation'] = orientation.astype(NUMPY_TYPE)
        obj['multibin_confidence'] = confidence.astype(NUMPY_TYPE)

        # add our implementation here
        obj['tricosine'] =  alpha_rad_to_tricoine(
            obj['new_alpha']).astype(NUMPY_TYPE)
        obj['alpha_sector'] = angle2cat(obj['new_alpha'],num_alpha_sectors)
        obj['rot_y_sector'] = angle2cat(obj['new_alpha'],num_rot_y_sectors)

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
        obj['multibin_orientation_flipped'] = orientation.astype(NUMPY_TYPE)
        obj['multibin_confidence_flipped'] = confidence.astype(NUMPY_TYPE)
        # add our implementation here
        obj['tricosine_flipped'] = alpha_rad_to_tricoine(
            2.*np.pi - obj['new_alpha']).astype(NUMPY_TYPE)
        obj['alpha_sector_flipped'] = angle2cat(2.*np.pi -obj['new_alpha'],num_alpha_sectors)
        obj['rot_y_sector_flipped'] = angle2cat(2.*np.pi -obj['new_alpha'],num_rot_y_sectors)
        center = ((obj['xmin']+obj['xmax'])/2,(obj['ymin']+obj['ymax'])/2)
        obj['view_angle'] = center[0]/NORM_W*VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X/2)
        obj['distr'] = qualityaware(obj['rot_y_sector'],num_rot_y_sectors)
        obj['distr_flipped'] = qualityaware(obj['rot_y_sector_flipped'],num_rot_y_sectors)
        obj['orient_flipped'] = orientation 
        obj['conf_flipped'] = confidence

    '''
    for obj in all_objs:
        assert isinstance(obj['class_name'], str)  # str name of the class of the object

        assert isinstance(obj['class_id'], int)  # int id of the class of the object

        assert isinstance(obj['image_path'], Path)  # path to image of the object

        assert isinstance(obj['xmin'], int)  # relative xmin position of the object bbox
        assert obj['xmin'] >= 0.0 #and obj['xmin'] <= 1.0 this is the crop pixel values

        assert isinstance(obj['ymin'], int)  # relative ymin position
        assert obj['ymin'] >= 0.0 #and obj['ymin'] <= 1.0

        assert isinstance(obj['xmax'], int)  # relative xmax position
        assert obj['xmax'] >= 0.0 #and obj['xmax'] <= 1.0

        assert isinstance(obj['ymax'], int)  # relative ymax position
        assert obj['ymax'] >= 0.0 #and obj['ymax'] <= 1.0

        assert isinstance(obj['alpha'], float)  # kitti orientation in alpha
        assert obj['alpha'] >= -math.pi and obj['alpha'] <= math.pi

        assert isinstance(obj['rot_y'], float)  # kitti orientation in rot_y
        assert obj['rot_y'] >= -math.pi and obj['rot_y'] <= math.pi

        # add our implementation here

        assert isinstance(obj['tricosine'], np.ndarray)  # kitti orientation in tricosine
        assert obj['tricosine'].shape == (3,)
        assert obj['tricosine'].dtype == np.dtype('float32')

        assert isinstance(obj['multibin_orientation'], np.ndarray)  # kitti orientation in multibin_orientation
        assert obj['multibin_orientation'].shape == (2,2)
        assert obj['multibin_orientation'].dtype == np.dtype('float32')

        assert isinstance(obj['multibin_confidence'], np.ndarray)  # kitti orientation in multibin_confidence
        assert obj['multibin_confidence'].shape == (2,)
        assert obj['multibin_confidence'].dtype == np.dtype('float32')

        assert isinstance(obj['alpha_sector'], np.ndarray)  # kitti orientation in alpha_sector
        assert obj['alpha_sector'].shape == (num_alpha_sectors,)
        assert obj['alpha_sector'].dtype == np.dtype('float32')

        assert isinstance(obj['rot_y_sector'], np.ndarray)  # kitti orientation in rot_y_sector
        assert obj['rot_y_sector'].shape == (num_rot_y_sectors,)
        assert obj['rot_y_sector'].dtype == np.dtype('float32')
    '''
    return all_objs

# get the bounding box,  values for the instance
# this automatically does flips
def prepare_input_and_output(image_dir:str, train_inst, style:str = 'multibin'):
    # Prepare image patch
    xmin = train_inst['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    img = cv2.imread(image_dir + str(train_inst['image_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # crop the image using the obj bounding box, deepcopy to prevent memory sharing
    img = copy.deepcopy(img[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)

    # re-color the image
    #img += np.random.randint(-2, 3, img.shape).astype('float32')
    #t  = [np.random.uniform()]
    #t += [np.random.uniform()]
    #t += [np.random.uniform()]
    #t = np.array(t)

    #img = img * (1 + t)
    #img = img / (255. * 2.)

    # flip the image by random chance
    flip = np.random.binomial(1, .5)
    # flip image horizonatally
    if flip > 0.5:
        img = cv2.flip(img, 1)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    # zero center the image values around these (avg?) RGB values
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    #img = img[:,:,::-1]

    # if the image crop is flipped also flip the orientation values
    if style == 'multibin':
        if flip > 0.5:
            return img, train_inst['dims'], train_inst['multibin_orientation_flipped'], train_inst['multibin_confidence_flipped']
        else:
            return img, train_inst['dims'], train_inst['multibin_orientation'], train_inst['multibin_confidence']
    elif style == 'alpha':
        if flip > 0.5:
            return img, train_inst['dims'], train_inst['new_alpha']
        else:
            return img, train_inst['dims'], math.tau-train_inst['new_alpha']
    elif style == 'rot_y':
        if flip > 0.5:
            return img, train_inst['dims'], train_inst['rot_y']
        else:
            return img, train_inst['dims'], math.tau-train_inst['rot_y']
    elif style == 'rot_y_sector' or style == 'alpha_sector' or style == 'tricosine':
        if flip > 0.5:
            return img, train_inst['dims'], train_inst[style]
        else:
            return img, train_inst['dims'], train_inst['%s_flipped'%style]
    else:
        raise Exception("No such orientation type: %s"%style)

def fp_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class KittiGenerator(Sequence):

    '''Creates A KittiGenerator Sequence
    Args:
        label_dir (str) : path to the directory with labels
        image_dir (str) : path to the image directory
        mode (str): tells whether to be in train or test mode
        batch_size (int) : tells batchsize to use
    '''
    # update to remove kwargs
    def __init__(self, label_dir:str,
                 image_dir:str, 
                 mode = "train", 
                 batch_size = 8,
                 orientation_type = "multibin",
                 sectors = 4):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self._sectors = sectors
        self.all_objs = parse_annotation(label_dir,image_dir,mode,self._sectors,self._sectors)
        self.mode = mode
        self.batch_size = batch_size
        
        if mode=='test':
            warnings.warn("testing mode has not been inplemented yet")
        if mode=='val':
            warnings.warn("validation mode has not been inplemented yet")
            
        self._clen = len(self.all_objs)  # number of objects
        self._keys = list(range(self._clen))  # list of all object ids
        np.random.shuffle(self._keys)
        self.epochs = 0
        self.orientation_type = orientation_type

    def __len__(self)->int:
        return len(self.all_objs) // self.batch_size

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size  # start of key index
        r_bound = l_bound + self.batch_size  # end of key index
        r_bound = r_bound if r_bound < self._clen else self._clen  # check for key index overflow
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))  # batch of images
        d_batch = np.zeros((r_bound - l_bound, 3))  # batch of dimensions

        '''# output specs
        if self.orientation_type == "tricosine"
            # return obj_img_batch, obj_dim_batch, obj_tricosine_batch
        if self.orientation_type == "alpha"
            # return obj_img_batch, obj_dim_batch, obj_alpha_batch
        if self.orientation_type == "rot_y"
            # return obj_img_batch, obj_dim_batch, obj_rot_y_batch
        if self.orientation_type == "alpha_sector"
            assert obj_alpha_sec_batch.shape() == (bs, num_alpha_sectors)
            # return obj_img_batch, obj_dim_batch, obj_alpha_sec_batch
        if self.orientation_type == "rot_y_sector"
            assert obj_rot_y_sec_batch.shape() == (bs, num_rot_y_sectors)
            # return obj_img_batch, obj_dim_batch, obj_rot_y_sec_batch
        if self.orientation_type == "multibin"
            assert obj_multibin_orientation_batch.shape() == (bs, 2, 2)
            assert obj_multibin_conf_batch.shape() == (bs, 2, 1)
            # return obj_img_batch, obj_dim_batch, obj_multibin_orientation_batch, obj_multibin_conf_batch
        '''
        if self.orientation_type == "multibin":
            # batch of confs for each bin
            c_batch = np.zeros((r_bound - l_bound, BIN))
            o_batch = np.zeros((r_bound - l_bound, BIN, 2))
            for currt_inst, key in enumerate(self._keys[l_bound:r_bound]):
                image, dimension, orientation, confidence = prepare_input_and_output(
                    self.image_dir, self.all_objs[key])
                x_batch[currt_inst, :] = image
                d_batch[currt_inst, :] = dimension
                o_batch[currt_inst, :] = orientation
                c_batch[currt_inst, :] = confidence
            return x_batch, d_batch, o_batch, c_batch
        elif self.orientation_type == "rot_y_sector" or self.orientation_type == "alpha_sector":
            s_batch = np.zeros((r_bound - l_bound, self._sectors))
            for currt_inst, key in enumerate(self._keys[l_bound:r_bound]):
                image,dimension,sector = prepare_input_and_output(self.image_dir,self.all_objs[key],self.orientation_type)
                x_batch[currt_inst, :] = image
                d_batch[currt_inst, :] = dimension
                s_batch[currt_inst, :] = sector
            return x_batch,d_batch,s_batch
        elif self.orientation_type =='tricosine':
            tc_batch = np.zeros((r_bound - l_bound, 3))
            for currt_inst, key in enumerate(self._keys[l_bound:r_bound]):
                image,dimension,tricos = prepare_input_and_output(self.image_dir,self.all_objs[key],self.orientation_type)
                x_batch[currt_inst, :] = image
                d_batch[currt_inst, :] = dimension
                tc_batch[currt_inst, :] = tricos
            return x_batch,d_batch,tc_batch
        elif self.orientation_type == "alpha" or self.orientation_type == 'rot_y':
            a_batch = np.zeros((r_bound - l_bound, 1))
            for currt_inst, key in enumerate(self._keys[l_bound:r_bound]):
                image,dimension,angle = prepare_input_and_output(self.image_dir,self.all_objs[key],self.orientation_type)
                x_batch[currt_inst, :] = image
                d_batch[currt_inst, :] = dimension
                a_batch[currt_inst, :] = angle
            return x_batch,d_batch,a_batch
        else:
            raise Exception("Invalid Orientation Type")
            

    def on_epoch_end(self):
        print("initializing next epoch")
        np.random.shuffle(self._keys)
        self.epochs+=1
        self._idx = 0

    def __str__(self):
        return "KittiDatagenerator:<size %d,image_dir:%s,label_dir:%s,epoch:%d>"%(len(self),self.image_dir,self.label_dir,self.epochs)

    
    def to_tfrecord(self, path:str = './records/')->str:
        writer_path = '%s%s-%s-.tfrec'%(path,self.mode,datetime.now().strftime('%Y%m%d%H%M%S'))
        with tf.io.TFRecordWriter(writer_path) as writer:
            for c,i in tqdm(enumerate(self)):
                inp,out = i
                print(inp)
                feature = {
                    'idx': int_feature(c),
                    'input':tf.data.Dataset.from_tensor_slices(inp),
                    'output':tf.data.Dataset.from_tensor_slices(out)
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())
        return writer_path
