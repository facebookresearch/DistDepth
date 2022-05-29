# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image

# # SimSIN camera parameters
# focalLength = 256
# centerX = 512.0
# centerY = 512.0
# baseline = 0.131202
# intWidth = 512
# intHeight = 512

# # UniSIN camera parameters
# focalLength = 522.2714
# centerX = 658.70
# centerY = 364.48
# baseline = 0.1200
# intWidth = 1280
# intHeight = 720

# Hypersim camera parameters
focalLength = 886.81
centerX = 512.0
centerY = 384.0 
baseline = 0.131202
intWidth = 1024
intHeight = 768

def correct_distance_to_depth(npyDistance):
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], focalLength, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * focalLength
    return npyDepth

def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = Image.open(rgb_file).transpose(Image.FLIP_LEFT_RIGHT)
    depth = np.load(depth_file).T
    depth = np.flip(depth, axis=0)

    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = -depth[u,v]
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

if __name__ == '__main__':

    save_folder = 'pc_generation'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    target = 'data/sample_pc/0000_depth.npy'
    rgb_filename = 'data/sample_pc/0000.jpg'
    ply_filename = 'data/sample_pc/0000_pc.ply'

    generate_pointcloud(rgb_filename, target, ply_filename)