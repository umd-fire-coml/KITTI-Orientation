import tensorflow as tf
import numpy as np
from orientation_converters import multibin_orientation_confidence_to_alpha, trisector_affinity_to_angle, angle_normed_to_angle_rad

# Stateful metric over the entire dataset. 
# Because metrics are evaluated for each batch during training and evaluation,
# this metric will keep track of average accuracy over the entire dataset, 
# not the average accuracy of each batch.
class OrientationAccuracy(tf.keras.metrics.Metric):

    # Create the state variables in __init__
    def __init__(self, orientation_type, name='orientation_accuracy', **kwargs):
        super(OrientationAccuracy, self).__init__(name=name, **kwargs)

        # internal state variables
        self.orientation_type = orientation_type
        self.num_pairs = tf.Variable(0)  # num of pairs of y_true, y_pred
        self.sum_accuracy = tf.Variable(0.)  # sum of accuracies for each pair of y_true, y_pred
        self.cur_accuracy = self.add_weight(name='oa', initializer='zeros')  # current state of accuracy

    def aos_convert_to_alpha(self, tensor):
        # if orientation type is already 'alpha' or 'rot_y', no need to change
        if self.orientation_type in ['rot_y','alpha']: 
            return tensor
        else: 
            return self.recursive_aos(tensor)
            

    def recursive_aos(self, tensor): # test this
        # recursively unpacks tensor until the tensor dimension is 1xN, then operates
        s = tensor.get_shape()
        if len(s) > 1: # expecting a (n x N) tensor
            # return tf.map_fn(self.recursive_aos, tensor)
            return tf.stack([self.recursive_aos(un_packed_tensor)
                             for un_packed_tensor in tf.unstack(tensor)]) # make sure stack does not REVERSE
        else:
            # expecting a (1 x N) tensor
            arr = tensor.numpy()
            arr_type = arr.dtype

            # print('\n\n self.orientation_type: {}\n\n'.format(self.orientation_type))
            # print('\n\n arr: {}\n\n'.format(arr))
            # val = multibin_orientation_confidence_to_alpha(arr[0],arr[1])
            if self.orientation_type == 'multibin':  val = multibin_orientation_confidence_to_alpha(arr[0],arr[1]) # (1x2) -> (1x0) shape
            if self.orientation_type == 'tricosine': val = trisector_affinity_to_angle(arr)# (1x3) -> (1x0) shape
            # if orientation_type == 'rot_y_sectors': val = sector2angle(arr,len(arr)) # (1xSectors) -> (1x0) shape
            # if orientation_type == 'alpha_sectors': val = sector2angle(arr,len(arr)) # (1xSectors) -> (1x0) shape
            val = angle_normed_to_angle_rad(val)
            val = np.asarray(val, dtype=arr_type)
            return tf.constant(val) 
    
    # Update the variables given y_true and y_pred in update_state()
    def update_state(self, y_true, y_pred, sample_weight=None):

        # convert to alphas using orientation_converters and calculate the batch_accuracies

        alpha_pred = self.aos_convert_to_alpha(y_pred)
        alpha_true = self.aos_convert_to_alpha(y_true)
        alpha_delta = alpha_true - alpha_pred
        normalized  = 0.5 * (tf.math.cos(alpha_delta) + 1)
        batch_accuracies = tf.math.reduce_mean(normalized)

        # update the cur_accuracy
        self.sum_accuracy = tf.reduce_sum(batch_accuracies) + self.sum_accuracy # convert to float
        self.num_pairs = tf.size(batch_accuracies) + self.num_pairs
        self.cur_accuracy = tf.math.divide(self.sum_accuracy, tf.cast(self.num_pairs, tf.float32))

    # Return the metric result in result()
    def result(self):  
        return self.cur_accuracy