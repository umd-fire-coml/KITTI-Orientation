from data_processing import tricosine_to_alpha_rad
# from data_processing import multibin_to_alpha_rad
# from data_processing import alpha_to_roty_rad

def convert_orientation_to_roty(orientation, input_orientation_type):
    if orientation_type == 'rotation_y':
        # don't change orientation output
        roty = orientation

    if orientation_type == 'tricosine':
        # fp values:
        # 3 cosine values -> alpha convertor is there -> make into rot_y
        alph = tricosine_to_alpha_rad(orientation)
        roty = alpha_to_roty_rad(alph)

    if orientation_type == 'multibin':
        alph = multibin_to_alpha_rad(orientation)
        roty = alpha_to_roty_rad(alph)

    if orientation_type == 'alpha':
        roty = alpha_to_roty_rad(orientation)

    return roty
