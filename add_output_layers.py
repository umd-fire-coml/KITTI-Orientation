import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import math as K

NUM_SECTORS = 2**3 # TODO import from ?
BIN_MULTIBIN = 2

def first_output_stage(backbone_layer, out_shape, out_layer_name=''):
    y = layers.Dense(256)(backbone_layer)
    y = layers.LeakyReLU(0.1)(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(out_shape[0]*out_shape[1], name=out_layer_name)(y)

    return y

def add_output_layers(orientation_output_type, backbone_layer):
    backbone_layer = layers.Flatten()(backbone_layer)

    # keras replicated implementation of multibin
    if orientation_output_type == 'multibin':
        o_layer = first_output_stage(backbone_layer, [BIN_MULTIBIN, 2])
        o_layer = layers.Reshape([BIN_MULTIBIN, 2])(o_layer)
        o_layer = layers.Lambda(lambda a: K.l2_normalize(a,axis=2), name='o_layer_output')(o_layer) # l2_normalization

        c_layer_pre = first_output_stage(backbone_layer, [BIN_MULTIBIN, 1])
        c_layer = layers.Softmax(name='c_layer_output')(c_layer_pre)
        return o_layer, c_layer_pre, c_layer
    
    elif (orientation_output_type == 'rot_y_sectors' or orientation_output_type == 'alpha_sectors'): 
        output_shape = [NUM_SECTORS, 1]
        o_layer = first_output_stage(backbone_layer, output_shape)
        o_layer = layers.Softmax(name=(orientation_output_type+"_output"))(o_layer)
        return o_layer

    # If not multibin -> output first stage with different shapes
    elif orientation_output_type == 'alpha': output_shape = [1, 1]
    elif orientation_output_type == 'rot_y': output_shape = [1, 1]
    elif orientation_output_type == 'tricosine': output_shape = [3, 1]
    else: raise NameError("Invalid orientation_output_type")

    o_layer = first_output_stage(backbone_layer, output_shape, out_layer_name=(orientation_output_type+"_output"))
    return o_layer