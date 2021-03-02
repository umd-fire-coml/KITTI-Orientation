import tensorflow as tf
from backbone_xception import Xception_model
from add_output_layers import add_output_layers
from tensorflow.keras import Input, Model
from loss_function import *
import data_processing as dp
import os
import argparse
import time
from datetime import datetime
from metrics import OrientationAccuracy

import tensorflow as tf
# set up tensorflow GPU
tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.8)
            # device_count = {'GPU': 1}
        )
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.compat.v1.enable_eager_execution()

# Processing argument
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument(dest='orientation', type=str,
                    help='Orientation Type of the model. Options are tricosine, alpha, rot_y, multibin')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                    help='Define the batch size for training. Default value is 8')
parser.add_argument('--weight_dir', dest='weight_dir', type=str, default='weights',
                    help='Relative path to save weights. Default path is weights')
parser.add_argument('--epoch', dest='num_epoch', type=int, default=100,
                    help='Number of epoch used for training. Default value is 100')
parser.add_argument('--kitti_dir', dest='kitti_dir', type=str, default='dataset',
                    help='path to kitti dataset directory. Its subdirectory should have training/ and testing/. Default path is dataset/')
parser.add_argument('--val_split', dest='val_split', type=float, default=0.2,
                    help='Fraction of the dataset used for validation. Default val_split is 0.2')

args = parser.parse_args()


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def loss_func(orientation):
    if orientation == 'tricosine':
        return loss_tricosine
    elif orientation == 'alpha':
        return loss_alpha
    elif orientation == 'rot_y':
        return loss_rot_y
    elif orientation == 'multibin':
        return loss_multibin
    else:
        raise Exception('Incorrect orientation type for loss function')

if __name__ == "__main__":
    BATCH_SIZE = args.batch_size
    NUM_EPOCH = args.num_epoch
    ORIENTATION = args.orientation
    KITTI_DIR = args.kitti_dir
    WEIGHT_DIR = args.weight_dir
    VAL_SPLIT = args.val_split
    if not os.path.isdir(KITTI_DIR):
        raise Exception('kitti_dir is not a directory.')
    if ORIENTATION not in ['tricosine', 'alpha', 'rot_y', 'multibin']:
        raise Exception('Invalid Orientation Type.')
    if not 0.0 <= VAL_SPLIT <= 1.0:
        raise Exception('Invalid val_split range between [0.0, 1.0]')
    if not os.path.isdir(WEIGHT_DIR):
        os.mkdir(WEIGHT_DIR)

    LABEL_DIR = os.path.join(KITTI_DIR, 'training/label_2/')
    IMG_DIR = os.path.join(KITTI_DIR, 'training/image_2/')

    # Generator config
    generator = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                  orientation_type=ORIENTATION, mode='train', val_split=VAL_SPLIT)
    validation = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                   orientation_type=ORIENTATION, mode='val', val_split=VAL_SPLIT)

    # Building Model
    inputs = Input(shape=(224, 224, 3))
    x = Xception_model(inputs, pooling='avg')
    x = add_output_layers(ORIENTATION, x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=loss_func(ORIENTATION), optimizer='adam',
                    metrics=[])
                #   metrics=[OrientationAccuracy(ORIENTATION)], run_eagerly=True)

    start_time = time.time()


    # log directory for weights, history, tensorboard
    log_dir = os.path.join(WEIGHT_DIR, ORIENTATION + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + str(int(start_time)))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # model callback config
    checkpoint_path = os.path.join(log_dir, 'epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}-orientation_accuracy-{orientation_accuracy:.4f}.h5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # tensorboard logs path
    tb_log_dir = os.path.join(log_dir, "logs/scalars/")
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir, histogram_freq=1)

    # early stop callback and accuracy callback
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=20)
    train_history = model.fit(x=generator, epochs=NUM_EPOCH, verbose=1,
                              validation_data=validation, callbacks=[tb_callback, cp_callback])

    # save training history as json file
    with open(os.path.join(WEIGHT_DIR, 'training_hist.json'), 'w') as f:
        f.write(str(train_history.history))

    print('Training Finished. Weights and history are saved under directory:', WEIGHT_DIR)
    print('Total training time is', timer(start_time, time.time()))