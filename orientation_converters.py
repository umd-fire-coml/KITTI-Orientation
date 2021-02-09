import numpy as np
import math

# trisector affinity constants
SECTORS = 3
SECTOR_WIDTH = math.tau / SECTORS
HALF_SECTOR_WIDTH = SECTOR_WIDTH / 2

def angle_rad_to_trisector_affinity(angle_rad):
    """Return a numpy array of trisector affinity values from an angle in radians
    
    Key Properties:
    - output represent the affinity value (cos distance) to the middle of 3 sectors
    - affinity increases if an angle moves towards the middle of each sector
    - affinity decreases if an angle moves away from the middle of each sector
    - if the angle is at the middle of a sector, output 1
    - if the angle is at the exact opposite of the sector center, output -1
    - affinity for each sector is within the range [0.0-1.0]
    """

    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % math.tau
    
    # output array
    trisector_affinity = np.empty(shape=(SECTORS,), dtype=np.float32)

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
    bounding_sector_affinity = math.cos(offset_from_bounding_sector_mid)
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
    left_sector_affinity = math.cos(offset_from_left_sector_mid)
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
    right_sector_affinity = math.cos(offset_from_right_sector_mid)
    # insert to output array
    trisector_affinity[right_sector_num] = right_sector_affinity

    return trisector_affinity

def trisector_affinity_to_angle_rad(trisector_affinity, allow_negative_pi=False):
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
    offset_from_bounding_sector_mid = math.acos(bounding_sector_affinity)
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
    offset_from_left_sector_mid = math.acos(left_sector_affinity)
    # get the predicted angle based on offset_from_left_sector_mid then wrap, if tau+1 then 1
    predicted_angle_from_left_sector_offset = (left_sector_mid + offset_from_left_sector_mid) % math.tau

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
    offset_from_right_sector_mid = math.acos(right_sector_affinity)
    # get the predicted angle based on offset_from_right_sector_mid then wrap, if -1 then tau-1
    predicted_angle_from_right_sector_offset = (right_sector_mid - offset_from_right_sector_mid) % math.tau

    # get the predicted angle from bounding sector (based on left right offset signals)
    if offset_from_left_sector_mid < offset_from_right_sector_mid:
        predicted_angle_from_bounding_sector_offset = l_angle_from_bounding_sector_offset
    else:
        predicted_angle_from_bounding_sector_offset = r_angle_from_bounding_sector_offset

    # calculuate the mean of predicted angles
    predicted_angles = [predicted_angle_from_left_sector_offset, 
                        predicted_angle_from_right_sector_offset, 
                        predicted_angle_from_bounding_sector_offset]
    sum_sin_predicted_angles = np.sum(np.sin(predicted_angles))
    sum_cos_predicted_angles = np.sum(np.cos(predicted_angles))
    mean_angle = np.arctan2(sum_sin_predicted_angles, sum_cos_predicted_angles)

    if allow_negative_pi:
        return mean_angle
    else:
        return mean_angle % math.tau

