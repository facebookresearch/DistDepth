# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)

class DistDepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DistDepth options")

        # EXECUTION mode
        self.parser.add_argument("--exe",
                                 type=str,
                                 help="execution option",
                                 default="eval_save",
                                 choices=["train", "eval_save", "eval_save_all", "eval_measure", "eval_measure-M"])

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "Habitat_sim"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="distdepth")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of ResNet layers",
                                 default=152,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset option",
                                 default="SimSIN",
                                 choices=["VA", "SimSIN", "UniSIN", "NYUv2"])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=256)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--thre",
                                 type=float,
                                 help="threshold for edge map",
                                 default=0.95)  
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=12.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--dist_wt",
                                 type=float,
                                 help="distillation loss weight",
                                 default=1.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=50)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-1)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=10)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=10)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose", "img"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        
        # Multi options
        self.parser.add_argument('--use_future_frame',
                                 action='store_true',
                                 help='If set, will also use a future frame in time for matching.')
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)
        self.parser.add_argument("--depth_binning",
                                 help="defines how the depth bins are constructed for the cost"
                                      "volume. 'linear' is uniformly sampled in depth space,"
                                      "'inverse' is uniformly sampled in inverse depth space",
                                 type=str,
                                 choices=['linear', 'inverse'],
                                 default='linear'),
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=96)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
