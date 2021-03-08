import tensorflow as tf
from add_output_layers import MULTIBIN_LAYER_OUTPUT_NAME, TRICOSINE_LAYER_OUTPUT_NAME, ALPHA_ROT_Y_LAYER_OUTPUT_NAME

def __loss_tricosine(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

loss_tricosine = {TRICOSINE_LAYER_OUTPUT_NAME: __loss_tricosine}

def __loss_alpha_rot_y(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

loss_alpha_rot_y = {ALPHA_ROT_Y_LAYER_OUTPUT_NAME: __loss_alpha_rot_y}

def __loss_multibin_orientation(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    loss = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors
    return tf.reduce_mean(loss)

def __loss_multibin_confidence(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

loss_multibin_weights = {'o_layer_output': 8.0, 'c_layer_output': 1.0}

def __loss_multibin_orientation_confidence(y_true, y_pred):
    loss_orientation = __loss_multibin_orientation(y_true[..., :2], y_pred[..., :2])
    loss_confidence = __loss_multibin_confidence(y_true[..., 2:], y_pred[..., 2:])
    return loss_multibin_weights['o_layer_output']*loss_orientation + loss_multibin_weights['c_layer_output']*loss_confidence

loss_multibin = {MULTIBIN_LAYER_OUTPUT_NAME: __loss_multibin_orientation_confidence}

def get_loss_function(orientation):
    if orientation == 'tricosine':
        return loss_tricosine
    elif orientation == 'alpha' or orientation == 'rot_y':
        return loss_alpha_rot_y
    elif orientation == 'multibin':
        return loss_multibin
    else:
        raise Exception('Incorrect orientation type for loss function')