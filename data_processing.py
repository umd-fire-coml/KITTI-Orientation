import tensorflow as tf
import cv2, os
import numpy as np
from random import shuffle
import copy

#####
#Training setting
BIN, OVERLAP = 2, 0.1
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram','Pedestrian','Cyclist']


def compute_anchors(angle):
    # angle is the new_alpha angle between 0 and 2pi
    anchors = []
    wedge = 2.*np.pi/BIN # angle size of each bin, i.e. 180 deg
    l_index = int(angle/wedge) # round down with int, tells me which bin the angle belongs to, gets to be either 0 or 1
    r_index = l_index + 1 # get to be either 1 or 2
    
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
        anchors.append([r_index%BIN, angle - r_index*wedge])
        
    return anchors

def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key:np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key:0 for key in VEHICLES}
        
    for label_file in sorted(os.listdir(label_dir)):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                # offset to make new_alpha, so that if car is head facing the camera, new_alpha = pi
                # , and if car is back facing the camera, new_alpha = 0
                new_alpha = float(line[3]) + np.pi/2. 
                # make new_alpha always >= 0
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                # make new_alpha always <= 2pi, equivalent to if new_alpha > 2.*np.pi: new_alpha = new_alpha - 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name':line[0],
                       'image':image_file,
                       'xmin':int(float(line[4])),
                       'ymin':int(float(line[5])),
                       'xmax':int(float(line[6])),
                       'ymax':int(float(line[7])),
                       'dims':np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                # get the new average
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data
                
    ###### flip data
    for obj in all_objs:
        
        # Get the dimensions offset from average (basically zero centering the values)
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((BIN,2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(obj['new_alpha'])

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Fix orientation and confidence for flip
        orientation = np.zeros((BIN,2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1
            
        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence
            
    return all_objs


def prepare_input_and_output(image_dir, train_inst):
    ### Prepare image patch
    xmin = train_inst['xmin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax'] #+ np.random.randint(-MAX_JIT, MAX_JIT+1)
    img = cv2.imread(image_dir + train_inst['image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = copy.deepcopy(img[ymin:ymax+1,xmin:xmax+1]).astype(np.float32)
    
    # re-color the image
    #img += np.random.randint(-2, 3, img.shape).astype('float32')
    #t  = [np.random.uniform()]
    #t += [np.random.uniform()]
    #t += [np.random.uniform()]
    #t = np.array(t)

    #img = img * (1 + t)
    #img = img / (255. * 2.)

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)
        
    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    #img = img[:,:,::-1]
    
    ### Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']

def data_gen(image_dir, all_objs, batch_size):
    num_obj = len(all_objs)
    
    keys = range(num_obj)
    np.random.shuffle(keys)
    
    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj
    
    while True:
        if l_bound == r_bound:
            l_bound  = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)
        
        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        c_batch = np.zeros((r_bound - l_bound, BIN))
        
        for key in keys[l_bound:r_bound]:
            # augment input image and fix object's orientation and confidence
            image, dimension, orientation, confidence = prepare_input_and_output(image_dir, all_objs[key])
            
            #plt.figure(figsize=(5,5))
            #plt.imshow(image/255./2.); plt.show()
            #print dimension
            #print orientation
            #print confidence
            
            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence
            
            currt_inst += 1
                
        yield x_batch, [d_batch, o_batch, c_batch]
        
        l_bound  = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj: r_bound = num_obj

