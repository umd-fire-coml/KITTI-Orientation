import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import math as K
from functools import reduce
from orientation_converters import ORIENTATION_SHAPE, CONFIDENCE_SHAPE, TRICOSINE_SHAPE, ALPHA_ROT_Y_SHAPE

MULTIBIN_LAYER_OUTPUT_NAME = 'multibin_layer_output'
TRICOSINE_LAYER_OUTPUT_NAME = 'tricosine_layer_output'
ALPHA_ROT_Y_LAYER_OUTPUT_NAME = 'alpha_rot_y_layer_output'

def add_dense_layers(backbone_layer, output_shape, out_layer_name=''):
    y = layers.Dense(256)(backbone_layer)
    y = layers.LeakyReLU(0.1)(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(reduce(lambda x, y: x*y, output_shape))(y)
    y = layers.Reshape(output_shape, name=out_layer_name)(y)
    return y

def add_output_layers(orientation_type, backbone_layer):
    backbone_layer = layers.Flatten()(backbone_layer)
    if orientation_type == 'multibin':
        o_layer = add_dense_layers(backbone_layer, ORIENTATION_SHAPE)
        o_layer = layers.Lambda(lambda a: K.l2_normalize(a,axis=2), name='o_layer_output')(o_layer) # l2_normalization

        c_layer = add_dense_layers(backbone_layer, CONFIDENCE_SHAPE, out_layer_name='c_layer_output')

        out_layer = layers.Concatenate(axis=-1, name=MULTIBIN_LAYER_OUTPUT_NAME, trainable=False)([o_layer, c_layer])
        # c_layer = layers.Softmax(name='c_layer_output')(c_layer_pre)
        # return o_layer, c_layer_pre, c_layer
        return out_layer
    elif orientation_type == 'tricosine':
        output_shape = TRICOSINE_SHAPE
        out_layer_name = TRICOSINE_LAYER_OUTPUT_NAME
    elif orientation_type == 'alpha' or orientation_type == 'rot_y': 
        output_shape = ALPHA_ROT_Y_SHAPE
        out_layer_name = ALPHA_ROT_Y_LAYER_OUTPUT_NAME
    else: raise NameError("Invalid orientation_output_type")

    out_layer = add_dense_layers(backbone_layer, output_shape, out_layer_name=out_layer_name)
    return out_layer