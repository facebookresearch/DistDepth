# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import os

TAR_DIR = 'AR_effects/results'

# image at background
IMG_IN = 'AR_effects/background.png'
DEPTH_IN = 'AR_effects/background_depth.npy'

# inserted object
OBJ_IM = 'AR_effects/obj_image.png'
OBJ_DEPTH = 'AR_effects/obj_depth.png'

# object resize factor
FAC = 0.25

# preset object size
DEPTH_MIN = 1.5
DEPTH_MAX = 2.0

os.makedirs(TAR_DIR, exist_ok=True)

back = cv2.imread(IMG_IN, -1)
depth = np.load(DEPTH_IN)

# crop out the ball region
object_img = cv2.imread(OBJ_IM,-1)[199:199+488, 705:705+488,:]
resize_res = (int(FAC*object_img.shape[0]), int(FAC*object_img.shape[1]))
object_img = cv2.resize(object_img, (resize_res[1], resize_res[0]))

# crop out the ball region in its depth map
obj_depth = cv2.imread(OBJ_DEPTH,-1)[199:199+488, 705:705+488,:]
obj_depth = cv2.resize(cv2.cvtColor(obj_depth, cv2.COLOR_BGR2GRAY), (resize_res[1], resize_res[0]))
alpha_map = obj_depth > 80.99
min_ = obj_depth[alpha_map].min()
max_ = obj_depth[alpha_map].max()

# insert start pos
place_start = (350, 650)


for l in range(150):
    back_c = back.copy()
    place_current = (place_start[0], place_start[1]-l) # dragging to left side
    for i in range(resize_res[0]):
        for j in range(resize_res[1]):
            if alpha_map[i,j]: 
                depth_obj = DEPTH_MIN + (DEPTH_MAX-DEPTH_MIN)*obj_depth[i,j]/(max_-min_)
                if depth[place_current[0]+i, place_current[1]+j] > depth_obj: # simple z-buffer
                    back_c[place_current[0]+i, place_current[1]+j, :] = object_img[i, j, :3]

    cv2.imwrite(f'{TAR_DIR}/{l:03d}.png', back_c)
