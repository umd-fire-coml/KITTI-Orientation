from quick_eval import quick_eval
from kitti_common import get_label_anno
import numpy as np

gt_dir = "./sample_gt.txt"

with open(gt_dir) as f:
    lines = f.read().splitlines()

"""

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""

# for line in lines:
# line = lines[0]
# line = line.split(',')
# line = np.array(line,dtype=np.float32) # convert all values to float

# print(line)

# gt_annot = {

#     'name':       int(line[0]),
#     'truncated':  line[1],
#     'occluded':   int(line[2]),
#     'alpha':      np.array([line[3]]), #????? why??
#     'bbox':       line[4:8],
#     'dimensions': line[8:11],
#     'location':   line[11:14],
#     'rotation_y': line[14]
# }

gt_annot = get_label_anno(gt_dir)

det1_annot = {k:v for k,v in gt_annot.items()}
det2_annot = {k:v for k,v in gt_annot.items()}
det3_annot = {k:v for k,v in gt_annot.items()}

det1_annot['dimensions'] *= 1.2   #[x*1.2  for x in det1_annot['dimensions']]
det2_annot['dimensions'] *= 1.08  #[x*1.08 for x in det2_annot['dimensions']]
det3_annot['dimensions'] *= 1.001 #[x*1.01 for x in det3_annot['dimensions']]

dets = [det1_annot,
        det2_annot,
        det3_annot]

gts = [gt_annot,
       gt_annot,
       gt_annot]

print(det1_annot['alpha'].shape[0])

res = quick_eval(gts,dets,list(gt_annot['name']))
print(res)