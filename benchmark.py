import data_processing as dp
import time
from tqdm import tqdm
path_to_labels = './training/label_2'
path_to_images = "./training/image_2"
t1 = time.time()
seq = dp.KittiGenerator(path_to_labels,path_to_images)
index = 0
for it in tqdm(range(8192)):
    ele = seq.__getitem__(index)
    length =len(ele)
    if length!=seq.batch_size:
        idx = 0
        continue
t2 = time.time()
print("For sequence takes %d time to complete"%(t2-t1))