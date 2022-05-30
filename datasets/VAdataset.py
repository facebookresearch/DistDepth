# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import os
import PIL.Image as pil

from .VAbase import VABase

class VADataset(VABase):
    def __init__(self, *args, **kwargs):
        super(VADataset, self).__init__(*args, **kwargs)

        # Normalized intrinsics: The first row is normalize by image_width,
        # the second row is normalized by image_height

        # not changed
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 640)

    def get_color(self, path, do_flip):
        
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        depth_path = os.path.join(path)

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt