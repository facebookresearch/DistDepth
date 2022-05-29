# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import PIL.Image as pil

from .NYUv2base import NYUv2Base

class NYUv2Dataset(NYUv2Base):
    def __init__(self, *args, **kwargs):
        super(NYUv2Dataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.8093, 0, 0.508, 0],
                           [0, 1.08125, 0.5286, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        width, height = 640, 480
        new_width, new_height = 608, 448
        self.left = (width - new_width)//2
        self.top = (height - new_height)//2
        self.right = (width + new_width)//2
        self.bottom = (height + new_height)//2

    def get_color(self, path, do_flip):
        color = self.loader(path)
        
        # center crop
        color = color.crop((self.left, self.top, self.right , self.bottom))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        depth_path = os.path.join(path)
        depth_gt = np.load(depth_path)
        depth_gt = depth_gt[self.top:self.bottom, self.left:self.right]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt