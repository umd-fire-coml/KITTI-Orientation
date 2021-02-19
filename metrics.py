import tensorflow as tf
from orientation_converters import multibin_orientation_confidence_to_alpha, trisector_affinity_to_angle, angle_normed_to_angle_rad

# Stateful metric over the entire dataset. 
# Because metrics are evaluated for each batch during training and evaluation,
# this metric will keep track of average accuracy over the entire dataset, 
# not the average accuracy of each batch.
class OrientationAccuracy(tf.keras.metrics.Metric):

    # Create the state variables in __init__
    def __init__(self, name='orientation_accuracy', orientation_type=None, **kwargs):
        super(OrientationAccuracy, self).__init__(name=name, **kwargs)

        # internal state variables
        self.orientation_type = orientation_type
        self.num_pairs = tf.Variable(0)  # num of pairs of y_true, y_pred
        self.sum_accuracy = tf.Variable(0.)  # sum of accuracies for each pair of y_true, y_pred
        self.cur_accuracy = self.add_weight(name='oa', initializer='zeros')  # current state of accuracy

    # Update the variables given y_true and y_pred in update_state()
    def update_state(self, y_true, y_pred, sample_weight=None):

        # convert to alphas using orientation_converters and calculate the batch_accuracies

        batch_accuracies = TODO

        # update the cur_accuracy
        self.sum_accuracy.assign_add(tf.reduce_sum(batch_accuracies))
        self.num_pairs.assign_add(tf.size(batch_accuracies))
        self.cur_accuracy.assign(tf.math.divide(self.sum_accuracy, self.num_pairs))

    # Return the metric result in result()
    def result(self):  
        return self.cur_accuracy

