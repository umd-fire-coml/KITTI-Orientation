    vector<tGroundtruth> gt   = loadGroundtruth(gt_dir + "/" + file_name,gt_success);
    vector<tDetection>   det  = loadDetections(result_dir +"/" + file_name,
            compute_aos, eval_image, eval_ground, eval_3d, det_success);