import tensorflow as tf
import numpy as np
from keras_metrics import aos_convert_to_alpha
from data_processing import tricosine_to_alpha_rad
from data_processing import sector2angle
from data_processing import multibin_to_alpha_rad  


tricosine_gt = [tricosine_to_alpha_rad([0.5,0.7,0.8]),
                tricosine_to_alpha_rad([0.3,0.2,0.6]),
                tricosine_to_alpha_rad([0.4,0.2,0.9])]

tricosine_in = tf.constant([[0.5,0.7,0.8],
                            [0.3,0.2,0.6],
                            [0.4,0.2,0.9]],dtype=tf.float64)

# sector_gt = [sector2angle([]),
#              sector2angle([0.3,0.2,0.6]),
#              sector2angle([0.4,0.2,0.9])]

multibin_gt = [multibin_to_alpha_rad([[[0.4,0.8],[-0.4,0.7]],[0.1,0.95]]),
               multibin_to_alpha_rad([[[0.5,0.6],[-0.2,0.4]],[0.9,0.8]]),
               multibin_to_alpha_rad([[[0.1,0.3],[0.4,0.2]],[0.7,0.4]])]

multibin_in = tf.constant([[[[0.4,0.8],[-0.4,0.7]],[0.1,0.95]],
                           [[[0.5,0.6],[-0.2,0.4]],[0.9,0.8]],
                           [[[0.1,0.3],[0.4,0.2]],[0.7,0.4]]],dtype=tf.float32)


tricosine_out = aos_convert_to_alpha(tricosine_in, 'tricosine')

print("Testing Tricosine...")
print("""
Results!:
---------
Input Tensor: {input}
Ground Truth: {gt}
Output Tensor: {output}
      """.format(input=tricosine_in,
                 output=tricosine_out,
                 gt=tricosine_gt))

multibin_out = aos_convert_to_alpha(multibin_in, 'multibin')

print("Testing Multibin...")
print("""
Results!:
---------
Input Tensor: {input}
Ground Truth: {gt}
Output Tensor: {output}
      """.format(input=multibin_in,
                 output=multibin_out,
                 gt=multibin_gt))
