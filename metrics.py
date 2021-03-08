import tensorflow as tf
import numpy as np
from add_output_layers import MULTIBIN_LAYER_OUTPUT_NAME, TRICOSINE_LAYER_OUTPUT_NAME, ALPHA_ROT_Y_LAYER_OUTPUT_NAME
from orientation_converters import (multibin_orientation_confidence_to_alpha,
                                    trisector_affinity_to_angle,
                                    angle_normed_to_angle_rad,
                                    MULTIBIN_SHAPE,
                                    TRICOSINE_SHAPE)

TF_TYPE = tf.float32

def get_metrics(orientation_type):
    if orientation_type == 'multibin':
        return {MULTIBIN_LAYER_OUTPUT_NAME: OrientationAccuracy(orientation_type)}
    elif orientation_type == 'tricosine':
        return {TRICOSINE_LAYER_OUTPUT_NAME: OrientationAccuracy(orientation_type)}
    elif orientation_type == 'alpha' or orientation_type == 'rot_y': 
        return {ALPHA_ROT_Y_LAYER_OUTPUT_NAME: OrientationAccuracy(orientation_type)}

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
        # sum of accuracies for each pair of y_true, y_pred
        self.sum_accuracy = tf.Variable(0.)
        self.cur_accuracy = self.add_weight(
            name='oa', initializer='zeros')  # current state of accuracy

    def aos_convert_to_alpha(self, tensor):
        # if orientation type is already 'alpha' or 'rot_y', no need to change
        if self.orientation_type in ['rot_y', 'alpha']:
            return angle_normed_to_angle_rad(tensor)
        else:
            return self.recursive_aos(tensor)

    @tf.autograph.experimental.do_not_convert
    def recursive_aos(self, tensor):  # test this
        # recursively unpacks tensor until the tensor dimension is same shape as orientation_converters
        tensor_shape = tensor.get_shape()
        if self.orientation_type == 'multibin':
            if tensor_shape == MULTIBIN_SHAPE:
                arr = tensor.numpy()
                alpha = multibin_orientation_confidence_to_alpha(
                    arr[..., :2], arr[..., 2:])
                return tf.constant(alpha, dtype=TF_TYPE)
            elif len(tensor_shape) > len(MULTIBIN_SHAPE):
                return tf.stack([self.recursive_aos(un_packed_tensor)
                                 for un_packed_tensor in tf.unstack(tensor)])
            else:
                raise Exception("multibin recursive_aos error")
        elif self.orientation_type == 'tricosine':
            if tensor_shape == TRICOSINE_SHAPE:
                arr = tensor.numpy()
                alpha = trisector_affinity_to_angle(arr)
                return tf.constant(alpha, dtype=TF_TYPE)
            elif len(tensor_shape) > len(TRICOSINE_SHAPE):
                return tf.stack([self.recursive_aos(un_packed_tensor)
                                 for un_packed_tensor in tf.unstack(tensor)])
            else:
                raise Exception("tricosine recursive_aos error")
        else:
            raise Exception("Invalid self.orientation_type: " +
                            self.orientation_type)

    # Update the variables given y_true and y_pred in update_state()
    def update_state(self, y_true, y_pred, sample_weight=None):

        # convert to alphas using orientation_converters and calculate the batch_accuracies
        alpha_pred = self.aos_convert_to_alpha(y_pred)
        alpha_true = self.aos_convert_to_alpha(y_true)
        alpha_delta = alpha_true - alpha_pred
        normalized = 0.5 * (tf.math.cos(alpha_delta) + 1)
        batch_accuracies = tf.math.reduce_mean(normalized)

        # update the cur_accuracy
        self.sum_accuracy = tf.reduce_sum(
            batch_accuracies) + self.sum_accuracy  # convert to float
        self.num_pairs = tf.size(batch_accuracies) + self.num_pairs
        self.cur_accuracy = tf.math.divide(
            self.sum_accuracy, tf.cast(self.num_pairs, tf.float32))

    # Return the metric result in result()
    def result(self):
        return self.cur_accuracy

    # Reset state
    def reset_states(self):
        self.num_pairs = tf.Variable(0)  # num of pairs of y_true, y_pred
        # sum of accuracies for each pair of y_true, y_pred
        self.sum_accuracy = tf.Variable(0.)
        self.cur_accuracy = self.add_weight(name='oa', initializer='zeros')  # current state of accuracy