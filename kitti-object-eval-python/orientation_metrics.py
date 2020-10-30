import tf.keras.metrics

# create a class for each type of orientation
class TricosineAccuracy(Metric):
  # given the predictions and gts
  # step 0: convert all to alpha
  # step 1: deltas = gts - predictions
  # step 2: accuracy = (1.0 + cos(deltas)) / 2.0
  # final step: return average(accuracy)
