import data_processing as dp
import time
from tqdm import tqdm
import logging

logging.basicConfig(filename = "outputs.log",format
                    = '%(asctime)s %(message)s',level
                    = logging.DEBUG)


path_to_labels = './training/label_2'
path_to_images = "./training/image_2"
batch_size = 8
logging.info("starting sequential method")
t1 = time.time()
seq = dp.KittiGenerator(path_to_labels,path_to_images)
index = 0
for it in tqdm(range(8192)):
    ele = seq.__getitem__(index)
    length =len(ele)
    logging.debug("%s"%ele)
    if length!=seq.batch_size:
        idx = 0
        continue
t2 = time.time()
print("For sequence takes %d time to complete"%(t2-t1))
logging.info("sequence took %d to complete"%(t2-t1))

logging.info("starting tf.dataset method")
t1 = time.time()
seq = dp.KittiGenerator(path_to_labels,path_to_images)
dataset = seq.get_tf_handle() 
index = 0
for it in tqdm(range(8192)):
    ele = dataset.batch(batch_size,drop_remainder = True)
    logging.debug("%s",ele)
t2 = time.time()
print("Dataset takes %d time to complete"%(t2-t1))
logging.info("dataset took %d complete"%(t2-t1))
