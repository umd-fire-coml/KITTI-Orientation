from data_processing import multibin_to_alpha_rad_test
from data_processing import compute_anchors
import math 
import numpy as np
from pprint import pprint

# alpha = [i*(np.pi/5) for i in range(0,10)] #0,20 works...
# alpha = [i/10 for i in range(0,60)]
# alpha = [i/4 for i in range(0,24)]
# alpha = [i/7 for i in range(0,42)]
alpha = [i/3 for i in range(0,18)]

anchors_list = [compute_anchors(a) for a in alpha]

BIN = 2
orientation = np.zeros((BIN, 2))
confidence = np.zeros(BIN)

pprint(anchors_list)

final_vals = []
for anchors in anchors_list:
    for anchor in anchors:
        try:
            # compute the cos and sin of the offset angles
            orientation[anchor[0]] = np.array(
                [np.cos(anchor[1]), np.sin(anchor[1])])
            # set confidence of the sector to 1
            confidence[anchor[0]] = 1.
        except Exception:
            # print(e)
            print('ERROR:',anchor)
    # if in both sectors, then each confidence is 1/2, this makes sure sum of confidence adds up to 1
    confidence = confidence / np.sum(confidence)

    final_vals.append(multibin_to_alpha_rad_test(orientation,confidence))

final_vals = [val%math.tau for val in final_vals]
offset_vals = [alpha[i] - final_vals[i] for i in range(0,len(final_vals))]

print("""

alpha_vals: {}

final_vals: {}

alpha_vals - final_vals: {}

""".format(alpha,
           final_vals,
           offset_vals))
