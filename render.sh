#!/bin/bash

# align the test camera pose (from sfm) to the model coordinate system
python scripts/postprocess_pose_sfm2ours.py --model_path ""

# render test camera views
python render.py --config configs/nvidia/balloon1.yaml --skip_train --iteration 30000