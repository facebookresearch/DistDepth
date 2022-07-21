#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python execute.py --exe eval_measure --log_dir='./tmp' --data_path VA --dataset VA  --batch_size 1 --load_weights_folder ckpts-distdepth-152-SimSIN-DPTLegacy --models_to_load encoder depth  --width 256 --height 256 --max_depth 10 --frame_ids 0 --num_layers 152
python execute.py --exe eval_measure --log_dir='./tmp' --data_path VA --dataset VA  --batch_size 1 --load_weights_folder ckpts-distdepth-152-SimSIN-DPTLarge --models_to_load encoder depth  --width 256 --height 256 --max_depth 10 --frame_ids 0 --num_layers 152
python execute.py --exe eval_measure-M --log_dir './tmp' --data_path VA --dataset VA --batch_size 1 --load_weights_folder ckpts-distdepth-M-101-SimSIN-DPTLegacy  --max_depth 10  --num_layers 101 --models_to_load encoder depth pose_encoder pose
