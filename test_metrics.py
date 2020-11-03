import tensorflow as tf
from keras_metrics import aos_convert_to_alpha

a = tf.constant([[0.5,0.7,0.8],[0.3,0.2,0.6],[0.4,0.2,0.9]])
aos_convert_to_alpha(a, 'tricosine')
