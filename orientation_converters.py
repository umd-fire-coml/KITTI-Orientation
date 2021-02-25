import numpy as np
'''All methods here assume a single element of input, not a numpy array of inputs'''

# global constants
TAU = np.pi * 2.

# trisector affinity constants
SECTORS = 3
SECTOR_WIDTH = TAU / SECTORS
HALF_SECTOR_WIDTH = SECTOR_WIDTH / 2

def angle_to_trisector_affinity(angle_rad):
    """Return a numpy array of trisector affinity values from an angle (such as alpha or rot_y) in radians
    
    Key Properties:
    - output represent the affinity value (cos distance) to the middle of 3 sectors
    - affinity increases if an angle moves towards the middle of each sector
    - affinity decreases if an angle moves away from the middle of each sector
    - if the angle is at the middle of a sector, output 1
    - if the angle is at the exact opposite of the sector center, output -1
    - affinity for each sector is within the range [0.0-1.0]
    """

    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU
    
    # output array
    trisector_affinity = np.empty(shape=(SECTORS,))

    # calculate the bounding sector affinity
    # get the bounding sector number in which the angle is within the bounds of sector's start and end
    bounding_sector_num = int(new_angle_rad // SECTOR_WIDTH)
    # get the bounding sector's start position
    bounding_sector_start = SECTOR_WIDTH * bounding_sector_num
    # get the bounding sector's mid position
    bounding_sector_mid = bounding_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the the middle of the bounding sector
    offset_from_bounding_sector_mid = new_angle_rad - bounding_sector_mid
    # get the sector affinity based on the offset
    bounding_sector_affinity = np.cos(offset_from_bounding_sector_mid)
    # insert to output array
    trisector_affinity[bounding_sector_num] = bounding_sector_affinity

    # calculate the left sector affinity
    # get the left sector num, simply minus 1 then wrap
    left_sector_num = (bounding_sector_num - 1) % SECTORS  # if -1 then we get 2
    # get the left sector's start position
    left_sector_start = SECTOR_WIDTH * left_sector_num
    # get the left sector's mid position
    left_sector_mid = left_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the start of the bounding sector
    offset_from_bounding_sector_start = new_angle_rad - bounding_sector_start
    # get how much is the angle offset from the the middle of the left sector
    offset_from_left_sector_mid = HALF_SECTOR_WIDTH + offset_from_bounding_sector_start
    # get the sector affinity based on the offset
    left_sector_affinity = np.cos(offset_from_left_sector_mid)
    # insert to output array
    trisector_affinity[left_sector_num] = left_sector_affinity

    # calculate the right sector affinity
    # get the right sector num, simply plus 1 then wrap
    right_sector_num = (bounding_sector_num + 1) % SECTORS # if 3 we get 0
    # get the right sector's start position
    right_sector_start = SECTOR_WIDTH * right_sector_num
    # get the right sector's mid position
    right_sector_mid = right_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the end of the bounding sector
    offset_from_bounding_sector_end = SECTOR_WIDTH - offset_from_bounding_sector_start
    # get how much is the angle offset from the the middle of the right sector
    offset_from_right_sector_mid = HALF_SECTOR_WIDTH + offset_from_bounding_sector_end
    # get the sector affinity based on the offset
    right_sector_affinity = np.cos(offset_from_right_sector_mid)
    # insert to output array
    trisector_affinity[right_sector_num] = right_sector_affinity

    return trisector_affinity

def trisector_affinity_to_angle(trisector_affinity, allow_negative_pi=True):
    """Return an angle in radians from trisector affinity, allow_negative_pi sets the output range [-pi to +pi]
    """

    # clip values between -1 and 1 for acos.
    trisector_affinity = np.clip(trisector_affinity, -1.0, 1.0)

    # calculate the possible angles based on bounding sector offset
    # get the bounding sector number
    bounding_sector_num = np.argmax(trisector_affinity)
    # get the bounding sector's start position
    bounding_sector_start = SECTOR_WIDTH * bounding_sector_num
    # get the bounding sector's mid position
    bounding_sector_mid = bounding_sector_start + HALF_SECTOR_WIDTH
    # get bounding sector affinity
    bounding_sector_affinity = trisector_affinity[bounding_sector_num]
    # get how much is the angle offset from the the middle of the bounding sector
    offset_from_bounding_sector_mid = np.arccos(bounding_sector_affinity)
    # get the two possible angles based on offset_from_bounding_sector_mid
    l_angle_from_bounding_sector_offset = bounding_sector_mid - offset_from_bounding_sector_mid
    r_angle_from_bounding_sector_offset = bounding_sector_mid + offset_from_bounding_sector_mid

    # calculate the possible angle based on left sector offset
    # get the left sector num, simply minus 1 then wrap
    left_sector_num = (bounding_sector_num - 1) % SECTORS  # if -1 then we get 2
    # get the left sector's start position
    left_sector_start = SECTOR_WIDTH * left_sector_num
    # get the left sector's mid position
    left_sector_mid = left_sector_start + HALF_SECTOR_WIDTH
    # get left sector affinity
    left_sector_affinity = trisector_affinity[left_sector_num]
    # get how much is the angle offset from the the middle of the left sector
    offset_from_left_sector_mid = np.arccos(left_sector_affinity)
    # get the predicted angle based on offset_from_left_sector_mid then wrap, if tau+1 then 1
    predicted_angle_from_left_sector_offset = (left_sector_mid + offset_from_left_sector_mid) % TAU

    # calculate the possible angle based on left sector offset
    # get the right sector num, simply plus 1 then wrap
    right_sector_num = (bounding_sector_num + 1) % SECTORS # if 3 we get 0
    # get the right sector's start position
    right_sector_start = SECTOR_WIDTH * right_sector_num
    # get the right sector's mid position
    right_sector_mid = right_sector_start + HALF_SECTOR_WIDTH
    # get right sector affinity
    right_sector_affinity = trisector_affinity[right_sector_num]
    # get how much is the angle offset from the the middle of the right sector
    offset_from_right_sector_mid = np.arccos(right_sector_affinity)
    # get the predicted angle based on offset_from_right_sector_mid then wrap, if -1 then tau-1
    predicted_angle_from_right_sector_offset = (right_sector_mid - offset_from_right_sector_mid) % TAU

    # get the predicted angle from bounding sector (based on left right offset signals)
    if offset_from_left_sector_mid < offset_from_right_sector_mid:
        predicted_angle_from_bounding_sector_offset = l_angle_from_bounding_sector_offset
    else:
        predicted_angle_from_bounding_sector_offset = r_angle_from_bounding_sector_offset

    # calculuate the mean of predicted angles
    predicted_angles = np.asarray([predicted_angle_from_left_sector_offset, 
                        predicted_angle_from_right_sector_offset, 
                        predicted_angle_from_bounding_sector_offset])
    sum_sin_predicted_angles = np.sum(np.sin(predicted_angles))
    sum_cos_predicted_angles = np.sum(np.cos(predicted_angles))
    mean_angle = np.arctan2(sum_sin_predicted_angles, sum_cos_predicted_angles)

    if allow_negative_pi:
        return mean_angle
    else:
        return mean_angle % TAU

# multibin constants
NUM_BIN = int(2)
OVERLAP = 0.1
WEDGE_SIZE = TAU / NUM_BIN  # angle size of each bin, i.e. 180 deg

def alpha_to_new_alpha(alpha):
    '''Returns new alpha for multibin anchors from kitti alpha'''
    # offset to make new_alpha, so that if car is head facing the camera, new_alpha = pi
    # , and if car is back facing the camera, new_alpha = 0
    new_alpha = alpha + np.pi / 2.
    # make new_alpha always >= 0
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
    # make new_alpha always <= 2pi, equivalent to if new_alpha > 2.*np.pi: new_alpha = new_alpha - 2.*np.pi
    new_alpha = new_alpha - int(new_alpha / TAU) * TAU
    return new_alpha

def new_alpha_to_alpha(new_alpha):
    alpha_with_right_offset = (new_alpha + np.pi) % TAU - np.pi
    return alpha_with_right_offset - (np.pi / 2.)

def new_alpha_to_anchors(new_alpha):
    # angle is the new_alpha angle between 0 and 2pi
    anchors = []

    # round down with int, tells me which bin the angle belongs to, gets to be either 0 or 1
    l_index = int(new_alpha / WEDGE_SIZE)
    r_index = l_index + 1  # get to be either 1 or 2

    # (angle - l_index*wedge) is the +offset angle from start of the wedge, l_index*wedge is either 0 or 180
    # wedge/2 * (1+OVERLAP/2) is 90 deg * 1.05
    # basically check if the angle is within majority part of the current wedge
    if (new_alpha - l_index * WEDGE_SIZE) < WEDGE_SIZE / 2 * (1 + OVERLAP / 2):
        # append the bin index of the angle, and the +offset angle from start of the wedge
        anchors.append([l_index, new_alpha - l_index * WEDGE_SIZE])

    # r_index*wedge - angle is the -offset angle from start of the next wedge, r_index*wedge is either 180 or 360 deg
    # wedge/2 * (1+OVERLAP/2) is 90 deg * 1.05
    # basically check if the angle is also within majority part of the next wedge
    if (r_index * WEDGE_SIZE - new_alpha) < WEDGE_SIZE / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % NUM_BIN, new_alpha - r_index * WEDGE_SIZE])
    return anchors

def alpha_to_anchors(alpha):
    return new_alpha_to_anchors(alpha_to_new_alpha(alpha))

def alpha_to_multibin_orientation_confidence(alpha):
    # Get orientation and confidence values
    # set all values as zeros for each orientation (2x2 values) and conf  (2 values, each value represents the sector)
    orientation = np.zeros((NUM_BIN, 2))
    confidence = np.zeros(NUM_BIN)

    anchors = alpha_to_anchors(alpha)

    # get the sector id and offset from center of each sector if its within +-94.5 deg from the center
    for anchor in anchors:
        # compute the cos and sin of the offset angles
        orientation[anchor[0]] = np.array(
            [np.cos(anchor[1]), np.sin(anchor[1])])
        # set confidence of the sector to 1
        confidence[anchor[0]] = 1.

    # if in both sectors, then each confidence is 1/2, this makes sure sum of confidence adds up to 1
    confidence = confidence / np.sum(confidence)

    return orientation, confidence

def multibin_orientation_confidence_to_new_alpha(orientation, confidence):

    new_alpha_confidences = []
    new_alpha_predictions = []

    for bin_index, bin_confidence in enumerate(confidence):
        cos = orientation[bin_index][0] #[bin_index, 0]
        sin = orientation[bin_index][1] #[bin_index, 1]
        new_alpha_minus_wedge_start = np.arctan2(sin, cos)
        wedge_start = bin_index * WEDGE_SIZE
        new_alpha_prediction = new_alpha_minus_wedge_start + wedge_start
        new_alpha_prediction = new_alpha_prediction % TAU
        new_alpha_confidences.append(bin_confidence)
        new_alpha_predictions.append(new_alpha_prediction)
    
    # compute the weighted average
    new_alpha = np.average(new_alpha_predictions, weights=new_alpha_confidences)
    return new_alpha

def multibin_orientation_confidence_to_alpha(orientation, confidence):
    return new_alpha_to_alpha(multibin_orientation_confidence_to_new_alpha(orientation, confidence))


# alpha and rot_y constants
ALPHA_ROT_Y_NORM_FACTOR = np.pi

def angle_to_angle_normed(angle_rad):
    '''normalize angle_rad to [-1,1]'''
    return angle_rad / ALPHA_ROT_Y_NORM_FACTOR

def angle_normed_to_angle_rad(angle_normed):
    return angle_normed * ALPHA_ROT_Y_NORM_FACTOR

def alpha_to_rot_y(alpha, loc_x, loc_z):
    return alpha + np.arctan(loc_x/loc_z)


def numpy_helper(tensor):
    return tensor.numpy()
