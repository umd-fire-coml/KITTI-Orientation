import tensorflow as tf
from add_output_layers import TRICOSINE_LAYER_OUTPUT_NAME, ALPHA_ROT_Y_LAYER_OUTPUT_NAME


def __loss_tricosine(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


loss_tricosine = {TRICOSINE_LAYER_OUTPUT_NAME: __loss_tricosine}
loss_tricosine_weights = {TRICOSINE_LAYER_OUTPUT_NAME: 1.0}


def __loss_alpha_rot_y(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

loss_alpha_rot_y = {ALPHA_ROT_Y_LAYER_OUTPUT_NAME: __loss_alpha_rot_y}
loss_alpha_rot_y_weights = {ALPHA_ROT_Y_LAYER_OUTPUT_NAME: 1.0}

def __loss_multibin_orientation(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), axis=1)
    loss = tf.add(tf.multiply(y_true[:, :, 0], y_pred[:, :, 0]),
                  tf.multiply(y_true[:, :, 1], y_pred[:, :, 1]))
    loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors
    return tf.reduce_mean(loss)


def __loss_multibin_confidence(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(y_true), logits=tf.squeeze(y_pred)))

@tf.autograph.experimental.do_not_convert
def __loss_multibin_output(y_true, y_pred):
    return tf.add(tf.multiply(loss_multibin_weights['o_layer_output'], __loss_multibin_orientation(y_true[..., :2], y_pred[..., :2])),
                  tf.multiply(loss_multibin_weights['c_layer_output'], __loss_multibin_confidence(y_true[..., 2:], y_pred[..., 2:])))

loss_multibin = {
    'o_layer_output': __loss_multibin_orientation,
    'c_layer_output': __loss_multibin_confidence
}

loss_multibin_weights = {
    'o_layer_output': 8.0/9.0,
    'c_layer_output': 1.0/9.0
}

def get_loss_params(orientation):
    if orientation == 'tricosine':
        return loss_tricosine, loss_tricosine_weights
    elif orientation == 'alpha' or orientation == 'rot_y':
        return loss_alpha_rot_y, loss_alpha_rot_y_weights
    elif orientation == 'multibin':
        return loss_multibin, loss_multibin_weights
    else:
        raise Exception('Incorrect orientation type for loss function')
