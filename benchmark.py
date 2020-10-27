#!/usr/bin/env python
## rip server is 3.5 :(
import data_processing as dp
import time
from tqdm import tqdm
import logging

logging.basicConfig(filename = "outputs.log",format
                    = '%(asctime)s %(message)s',level
                    = logging.DEBUG)
path_to_labels = './dataset/training/label_2/'
path_to_images = "./dataset/training/image_2/"
batch_size = 8
cache = 8
logging.info("starting sequential method")
t1 = time.time()
seq = dp.KittiGenerator(path_to_labels,path_to_images)
test_size = 64
for idx,ele in tqdm(enumerate(seq)):
    logging.debug("%d: %d %s"%(idx,len(ele),str(ele[0].shape)))
    if idx>test_size:
        break
t2 = time.time()
logging.debug("sequence took %d to complete"%(t2-t1))
'''
logging.info("starting tf.dataset method")
t1 = time.time()
seq = dp.KittiGenerator(path_to_labels,path_to_images)
dataset = seq.get_tf_handle() 
index = 0
for it in tqdm(range(int(test_size))):
    ele = dataset.batch(cache,drop_remainder = True)
    print(list(dataset.as_numpy_iterator()))
t2 = time.time()
logging.debug("dataset took %d complete"%(t2-t1))
'''