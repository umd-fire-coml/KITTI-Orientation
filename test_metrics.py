import tensorflow as tf
from keras_metrics import aos_convert_to_alpha
from data_processing import tricosine_to_alpha_rad

gt = [tricosine_to_alpha_rad([0.5,0.7,0.8]),
      tricosine_to_alpha_rad([0.3,0.2,0.6]),
      tricosine_to_alpha_rad([0.4,0.2,0.9])]

a = tf.constant([[0.5,0.7,0.8],
                 [0.3,0.2,0.6],
                 [0.4,0.2,0.9]],dtype=tf.float64)
                 
b = aos_convert_to_alpha(a, 'tricosine')

print(gt)
print(a)
print(b)