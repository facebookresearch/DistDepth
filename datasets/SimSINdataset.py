# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import cv2
import h5py
import numpy as np
import os
import PIL.Image as pil
import skimage.transform

from .SimSINbase import SimSINBase

class SimSINDataset(SimSINBase):
    def __init__(self, *args, **kwargs):
        super(SimSINDataset, self).__init__(*args, **kwargs)

        # Normalized intrinsics: The first row is normalize by image_width,
        # the second row is normalized by image_height
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (512, 512)

    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        depth_path = os.path.join(path)
        depth_gt = np.load(depth_path)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt