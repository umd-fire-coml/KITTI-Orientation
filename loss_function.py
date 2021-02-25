import tensorflow as tf

def loss_tricosine(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def loss_alpha(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def loss_rot_y(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def loss_alpha_sector(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def loss_rot_y_sector(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def loss_multibin_orientation(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    loss = (y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])
    loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors
    return tf.reduce_mean(loss)

def loss_multibin_confidence(c_label, confidence):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits=confidence))

loss_multibin = {'c_layer_output': loss_multibin_confidence,
                 'o_layer_output': loss_multibin_orientation}

loss_weights = {'o_layer_output': 8.0,
                'c_layer_output': 1.0}
