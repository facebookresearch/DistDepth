# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import PIL.Image as pil

from .UniSINbase import UniSINBase
from IO import read

class UniSINDataset(UniSINBase):
    def __init__(self, *args, **kwargs):
        super(UniSINDataset, self).__init__(*args, **kwargs)

        # Normalized intrinsics: The first row is normalize by image_width,
        # the second row is normalized by image_height
        self.K = np.array([[0.408024, 0, 0.5146106, 0],
                           [0, 0.725377, 0.5062255, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1280, 720)

    def get_color(self, path, do_flip):
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        depth_path = os.path.join(path)
        depth_gt = read(depth_path)
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt