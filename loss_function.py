import tensorflow as tf

def loss_tricosine(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
def loss_alpha(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
def loss_rot_y(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)
def loss_alpha_sec(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
def loss_rot_y_sec(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
