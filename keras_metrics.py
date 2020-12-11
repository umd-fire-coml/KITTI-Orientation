import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from data_processing import tricosine_to_alpha_rad
from data_processing import sector2angle  
from data_processing import multibin_to_alpha_rad  

"""
Usage:
------
model.compile(loss='...', optimizer='...', metrics=[kitti_aos(orientation_type)])
model.fit(...)
"""

def kitti_aos(orientation_type):
    def metric(y_true, y_pred):
        alpha_pred = aos_convert_to_alpha(y_pred, orientation_type)
        alpha_true = aos_convert_to_alpha(y_true, orientation_type) # needed? or y_true (gt) always alpha?
        alpha_delta = alpha_true - alpha_pred
        normalized  = 0.5 * K.cos(alpha_delta + 1) 
        val         = K.sum(normalized)

        return val
    return metric

def aos_convert_to_alpha(tensor, orientation_type):
    if orientation_type == 'tricosine':     return aos_orientation_to_alpha_rad(tensor, 'tricosine')
    if orientation_type == 'rot_y_sectors': return aos_orientation_to_alpha_rad(tensor, 'rot_y_sectors')
    if orientation_type == 'alpha_sectors': return aos_orientation_to_alpha_rad(tensor, 'alpha_sectors')
    if orientation_type == 'multibin':      return aos_orientation_to_alpha_rad(tensor, 'multibin')
    # if orientation_type == 'rot_y':         return K.map_fun(roty_to_alpha_rad, tensor)  # OLD
    else:
        # if orientation type is already 'alpha' or 'rot_y', no need to change
        return tensor

def aos_orientation_to_alpha_rad(tensor, orientation_type):
    def recursive_aos(tensor): # test this
        # recursively unpacks tensor until the tensor dimension is 1, then operates

        if K.ndim(tensor) > 1: # (1 for (1xN) shape)
            return tf.map_fn(recursive_aos, tensor)
            # return tf.stack([aos_orientation_to_alpha_rad(un_packed_tensor, orientation_type) for un_packed_tensor in tf.unstack(tensor)]) # make sure stack does not REVERSE
        else:
            # expecting a (1 x N) tensor
            arr = tensor.numpy()
            arr_type = arr.dtype
            if orientation_type == 'multibin':      val = multibin_to_alpha_rad(arr) # (1x2) -> (1x0) shape
            if orientation_type == 'tricosine':     val = tricosine_to_alpha_rad(arr) # (1x3) -> (1x0) shape
            if orientation_type == 'rot_y_sectors': val = sector2angle(arr,len(arr)) # (1xSectors) -> (1x0) shape
            if orientation_type == 'alpha_sectors': val = sector2angle(arr,len(arr)) # (1xSectors) -> (1x0) shape
            val = np.asarray(val, dtype=arr_type)
            return tf.constant(val)
    return recursive_aos(tensor)

def aos_roty_to_alpha_rad(tensor):
    # recursively calls until arrays of 0 dimension (single values) are found, then operates 

    if K.ndim(tensor) > 0: # (0 for (1x0) shape)
        return tf.map_fn(aos_roty_to_alpha_rad, tensor)
        # return tf.stack([aos_roty_to_alpha_rad(tensor, orientation_type) for tensor in tf.unstack(tensor)])
    else:
        # expecting a (1 x 0) tensor of 1 rot_y value
        arr = np.asarray(tensor)
        val = sector2angle(arr,len(arr))
        return tf.constant(val)

"""
This works!!!
Model code after THIS!
---------------------
def my_elementwise_func(x):
    # expects an array
    return 3 + K.sum(x)

def recursive_map(inputs, str_in):
    def rec_map(inputs):
        print(K.ndim(inputs), inputs, str_in) # printouts @K.ndim>0 below
        if K.ndim(inputs) > 1:# change to fit each type of orientation (1 for tricosine?)
            return K.map_fn(rec_map,inputs)
            # return tf.stack([recursive_map(inputs, str_in) for inputs in tf.unstack(inputs)],axis=0)
        else:
            return my_elementwise_func(inputs)
    return rec_map(inputs)

inputs = tf.stack([2*K.eye(4), 3*K.eye(4), 4*K.eye(4)])
result = recursive_map(inputs, 69)  

---------------------------
--- PRINTOUTS --- BELOW ---
---------------------------

format: #DIMs TENSOR

3 tf.Tensor(
    [[[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]

    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]

    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]], shape=(3, 4, 4), dtype=float32)

2 tf.Tensor(
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

1 tf.Tensor([1. 0. 0. 0.], shape=(4,), dtype=float32)

0 tf.Tensor(1.0, shape=(), dtype=float32)
"""