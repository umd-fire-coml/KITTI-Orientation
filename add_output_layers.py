import tensorflow as tf
from tf.keras import layers


def first_output_stage(backbone_layer, out_shape):
    y = layers.Dense(256)(backbone_layer)
    y = layers.LeakyReLU(0.1)(y)
    y = layers.Dropout(0.5)(y)
    y = layers.Dense(out_shape[0]*out_shape[1])(y)

    return y

def add_output_layers(orientation_output_type, backbone_layer):
    
    # keras replicated implementation of multibin
    if orientation_output_type == 'multibin':
        o_layer = first_output_stage(backbone_layer, [BIN, 2])
        o_layer = layers.Dense(BIN*2)(o_layer)
        o_layer = layers.Reshape([BIN, 2])(o_layer)
        o_layer = Lambda(lambda a: K.l2_normalize(a,axis=2))(o_layer) # l2_normalization

        c_layer = first_output_stage(backbone_layer, [BIN, 1])
        return o_layer, c_layer

    # If not multibin -> output first stage with different shapes
    output_shape = [0,0]
    if orientation_output_type == 'alpha': output_shape = [1, 1]
    if orientation_output_type == 'rot_y': output_shape = [1, 1]
    if orientation_output_type == 'tricosine': output_shape = [TRICOSINE_BINS, 1]
    if orientation_output_type == 'rot_y_sectors': output_shape = [NUM_SECTORS, 1]
    if orientation_output_type == 'alpha_sectors': output_shape = [NUM_SECTORS, 1]
    
    o_layer = first_output_stage(backbone_layer, output_shape)
    return o_layer