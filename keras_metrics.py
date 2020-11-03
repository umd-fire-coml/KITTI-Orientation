from tensorflow import math as M
from tensorflow.keras import backend as K
from data_processing import alpha_sectors_to_alpha_rad  # <- Not not created
from data_processing import roty_sectors_to_alpha_rad   # <- Not not created
from data_processing import tricosine_to_alpha_rad
from data_processing import roty_to_alpha_rad           # <- Not not created

"""
Usage:
------
model.compile(loss='...', optimizer='...', metrics=[kitti_aos(orientation_type)])
model.fit(...)
"""

def kitti_aos(orientation_type):
    def metric(y_true, y_pred):
        alpha_pred = aos_convert_to_alpha(y_pred, orientation_type)
        alpha_true = aos_convert_to_alpha(y_true, orientation_type)

        val = M.cos((alpha_true - alpha_pred) + 1) # Can also use K
        val = M.scalar_mul(0.5, val)        # Can I use K???
        val = K.sum(val)
        #return aos_convert_to_alpha(val, orientation_type)
        return val
    return metric
        
def aos_convert_to_alpha(tensor, orientation_type):
    if orientation_type == 'tricosine':
        return K.map_fn(tricosine_to_alpha_rad, tensor) # map_fn NEEDS testing

    elif orientation_type == 'rot_y':
        return K.map_fun(roty_to_alpha_rad, tensor)

    elif orientation_type == 'rot_y_sectors':
        return K.map_fun(roty_sectors_to_alpha_rad, tensor)

    elif orientation_type == 'alpha_sectors':
        return K.map_fun(alpha_sectors_to_alpha_rad, tensor)
    
    # if orientation type is already alpha, no need to change