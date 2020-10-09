import tensorflow as tf
def loss(loss_type,y_true, y_pred):
    if loss_type == 'tricosine' or loss_type == 'alpha' or loss_type == 'rotation_y':
        return  tf.keras.losses.mean_squared_error(y_true, y_pred)
    elif loss_type == 'alpha_sector' or loss_type == 'rotation_y_sector':
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    else:
        raise Exception("Incorrect loss function type")

def tricosine_loss(y_true, y_pred):
    return  tf.keras.losses.mean_squared_error(y_true, y_pred)