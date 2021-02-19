import tensorflow as tf
import numpy as np

"""
Usage:
------
model.compile(loss='...', optimizer='...', metrics=[kitti_aos(orientation_type)])
model.fit(...)
"""

def get_kitti_metric(orientation_type):
    def metric(y_true, y_pred):
        alpha_pred = batch_convert_to_alpha(y_pred, orientation_type)
        alpha_true = batch_convert_to_alpha(y_true, orientation_type) # needed? or y_true (gt) always alpha?
        alpha_delta = alpha_true - alpha_pred
        normalized  = 0.5 * (tf.math.cos(alpha_delta) + 1)
        mean_offset = tf.math.reduce_mean(normalized)
        return mean_offset
    return metric

def batch_convert_to_alpha(tensor, orientation_type):
    if orientation_type == 'multibin':      return aos_orientation_to_alpha_rad(tensor, 'multibin')
    if orientation_type == 'tricosine':     return aos_orientation_to_alpha_rad(tensor, 'tricosine')
    if orientation_type == 'alpha':         return aos_orientation_to_alpha_rad(tensor, 'alpha')
    if orientation_type == 'rot_y':         return aos_orientation_to_alpha_rad(tensor, 'rot_y')
    else:
        raise Exception("No such orientation type: %s" % orientation_type)

def aos_orientation_to_alpha_rad(tensor, orientation_type):
    def recursive_aos(tensor): # test this
        # recursively unpacks tensor until the tensor dimension is 1xN, then operates
        s = tensor.get_shape()
        if s[0] > 1: # expecting a (n x N) tensor
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
