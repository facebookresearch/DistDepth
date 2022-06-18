# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} :{avg' + self.fmt + '}\n'
        return fmtstr.format(**self.__dict__)

def write_turbo_depth_metric(path, toplot, vmin=0.001, vmax=10.0):
    v_min = vmin
    v_max = vmax
    normalizer = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='turbo')
    colormapped_im = (mapper.to_rgba(toplot)[:,:,:3]*255).astype(np.uint8)
    cv2.imwrite(path, colormapped_im[:,:,[2,1,0]])

def output_to_depth(level, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction"""
    min_out = 1 / max_depth
    max_out = 1 / min_depth
    scaled_out = min_out + (max_out - min_out) * level
    depth = 1.312 / scaled_out
    return depth

def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def normalize_image(x):
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def sec_to_hm(t):
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)