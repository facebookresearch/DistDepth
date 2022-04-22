# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def output_to_depth(level, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction"""
    min_out = 1 / max_depth
    max_out = 1 / min_depth
    scaled_out = min_out + (max_out - min_out) * level
    depth = 1.2 / scaled_out
    return depth
