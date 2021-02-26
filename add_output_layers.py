import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import math as K
from functools import reduce
from orientation_converters import ORIENTATION_SHAPE, CONFIDENCE_SHAPE, TRICOSINE_SHAPE, ALPHA_ROT_Y_SHAPE

def add_dense_layers(backbone_layer, output_shape, out_layer_name=''):
    y = layers.Dense(256)(backbone_layer)
    y = layers.LeakyReLU(0.1)(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(reduce(lambda x, y: x*y, output_shape))(y)
    y = layers.Reshape(output_shape, name=out_layer_name)(y)
    return y

def add_output_layers(orientation_output_type, backbone_layer):
    backbone_layer = layers.Flatten()(backbone_layer)
    # keras replicated implementation of multibin
    if orientation_output_type == 'multibin':
        o_layer = add_dense_layers(backbone_layer, ORIENTATION_SHAPE)
        o_layer = layers.Lambda(lambda a: K.l2_normalize(a,axis=2), name='o_layer_output')(o_layer) # l2_normalization

        c_layer = add_dense_layers(backbone_layer, CONFIDENCE_SHAPE, out_layer_name='c_layer_output')
        # c_layer = layers.Softmax(name='c_layer_output')(c_layer_pre)
        # return o_layer, c_layer_pre, c_layer
        return o_layer, c_layer
    # If not multibin -> output first stage with different shapes
    elif orientation_output_type == 'alpha': output_shape = ALPHA_ROT_Y_SHAPE
    elif orientation_output_type == 'rot_y': output_shape = ALPHA_ROT_Y_SHAPE
    elif orientation_output_type == 'tricosine': output_shape = TRICOSINE_SHAPE
    else: raise NameError("Invalid orientation_output_type")

    o_layer = add_dense_layers(backbone_layer, output_shape, out_layer_name=(orientation_output_type+"_output"))
    return o_layer