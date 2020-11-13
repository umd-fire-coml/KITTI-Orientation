import tensorflow.keras as keras
import tensorflow as tf
from backbone_xception import Xception_model
from add_output_layers import add_output_layers
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
from loss_function import *
import numpy as np
import data_processing as dp
import os, argparse, time, pickle

# Processing argument
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument(dest = 'orientation', type = str, 
                   help = 'Orientation Type of the model. Options are tricosine, alpha, rot_y, alpha_sector, rot_y_sector, multibin')
parser.add_argument('--num_sector', dest = 'NUM_SECTOR', type = int, default = 4,
                   help = 'Number of sector for orientation type alpha_sector, rot_y_sector, and multibin. Default value is 4')
parser.add_argument('--num_multibin', dest= 'NUM_MULTIBIN', type= int, default = 2,
                    help = 'Number of bins for multibin type. Default value is 2')
parser.add_argument('--batch_size', dest = 'BATCH_SIZE', type = int, default = 8,
                   help = 'Define the batch size for training. Default value is 8')
parser.add_argument('--weight_dir', dest = 'weight_dir', type = str,
                   help = 'Relative path to save weights. Default path is weights')
parser.add_argument('--epoch', dest = 'num_epoch', type = int, default=100,
                    help = 'Number of epoch used for training. Default value is 100')
parser.add_argument('--kitti_dir', dest = 'kitti_dir', type = str, default='dataset',
                    help = 'path to kitti dataset directory. Its subdirectory should have training/ and testing/. Default path is dataset/')
parser.add_argument('--load_weight_dir', dest = 'load_weight_dir', type = str, default=None,
                    help = 'path to load weights from. Default is None')

args = parser.parse_args()

# data directory
data_label = os.path.join(args.kitti_dir, 'training/label_2/')
data_img = os.path.join(args.kitti_dir, 'training/image_2/')

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def loss_func(orientation):
    if orientation == 'tricosine':
        return loss_tricosine
    elif orientation == 'alpha':
        return loss_alpha
    elif orientation == 'rot_y':
        return loss_rot_y
    elif orientation == 'alpha_sector':
        return loss_alpha_sector
    elif orientation == 'rot_y_sector':
        return loss_rot_y_sector
    elif orientation == 'multibin':
        return loss_multibin
    else:
        raise Exception('Incorrect orientation type for loss function')


if __name__=="__main__":
    BATCH_SIZE = args.BATCH_SIZE 
    NUM_SECTOR = args.NUM_SECTOR
    NUM_BIN = args.NUM_MULTIBIN
    num_epoch = args.num_epoch
    orientation = args.orientation
    if args.weight_dir == None:
        weight_dir = 'weights'
        if not os.path.isdir(weight_dir):
            os.mkdir(weight_dir)
    else:
        weight_dir = args.weight_dir
    if not os.path.isdir(weight_dir):
        error_msg = 'Weight directory ['+ weight_dir+ '] does not exist'
        raise Exception(error_msg)
    if orientation not in ['tricosine', 'alpha', 'rot_y', 'alpha_sector', 'rot_y_sector', 'multibin']:
        raise Exception('Wrong Orientation Type. Use python training.py -h to see more')
        
    
    print('Training on orientation type:[' ,args.orientation,'] with batch_size =',BATCH_SIZE, 'and num_sector is =',NUM_SECTOR)    
    # Generator config
    print('Loading generator')
    generator = dp.KittiGenerator(label_dir= data_label, image_dir= data_img,sectors = NUM_SECTOR, batch_size = BATCH_SIZE,orientation_type = args.orientation )
    # Model callback config
    checkpoint_path = os.path.join(weight_dir, str(orientation) +'_model.{epoch:02d}.h5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

    # Building Model
    inputs = Input(shape=(224, 224, 3))
    x = Xception_model(inputs, pooling='avg')
    x = add_output_layers(orientation, x, NUM_SECTOR,NUM_BIN )
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=loss_func(orientation), optimizer='adam')
    if args.load_weight_dir:
        print("Loading weights from: {}".format(args.load_weight_dir))
        model.load_weights(args.load_weight_dir)
        initial_epoch = int(args.load_weight_dir.split(".")[-2])
    else:
        initial_epoch = 1
    print('Starting Training')
    start_time = time.time()
    history = model.fit(x = generator, epochs = num_epoch, verbose = 1,callbacks=[cp_callback], initial_epoch=initial_epoch)

    with open(os.path.join(weight_dir, 'training_hist.txt'), 'w') as f:
        f.write(str(history.history))
    print('Training Finished. Weights are saved under directory:', weight_dir)
    print('Total training time is', timer(start_time, time.time()))