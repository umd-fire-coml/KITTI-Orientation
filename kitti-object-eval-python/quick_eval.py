import kitti_common as kitti
from eval import get_official_eval_result


"""
Assuming list of lists containing values in the following format:

annotations = 
{
    'name': [],
    'truncated': [],
    'occluded': [],
    'alpha': [],
    'bbox': [],
    'dimensions': [],
    'location': [],
    'rotation_y': []
}

"""
def quick_eval(gts,
               dets,
               curr_cls):

    # HERE convert from model output to output needed for get_official_eval_result()
    gt_annos = gts
    dt_annos = dets
    current_class = curr_cls

    return get_official_eval_result(gt_annos, dt_annos, current_class)




