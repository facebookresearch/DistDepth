# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import glob
import os

import habitat_sim

cv2 = None

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# Helper function to render observations from the stereo agent
def _render(sim, display, depth=False, store_path=None, last=False):
    for idx in range(50):
        
        sx = random.randint(1, 7)
        # TODO: You can design your action space
        if sx == 1:
            obs = sim.step("turn_right")
        elif sx == 2:
            obs = sim.step("turn_right")
            obs = sim.step("move_forward")
        elif sx ==3:
            obs = sim.step("move_forward")
        elif sx ==4:
            obs = sim.step("move_forward")
            obs = sim.step("move_forward")
        elif sx == 5:
            obs = sim.step("move_forward")
            obs = sim.step("turn_left")
        elif sx == 6:
            obs = sim.step("turn_left")
            obs = sim.step("move_forward")
        elif sx == 7:
            obs = sim.step("move_forward")
            obs = sim.step("turn_right")

        
        if depth:
            per_20_left = np.percentile(obs["left_depth_sensor"], 20)
            per_50_left = np.percentile(obs["left_depth_sensor"], 50)
            per_80_left = np.percentile(obs["left_depth_sensor"], 80)
            per_20_right = np.percentile(obs["right_depth_sensor"], 20)
            per_50_right = np.percentile(obs["right_depth_sensor"], 50)
            per_80_right = np.percentile(obs["right_depth_sensor"], 80)
            if not last:
                # some very ad-hoc conditions to prevent too close or get stucking at the corners
                if (per_50_left < 0.2) or (per_50_right < 0.2) or (per_20_right < 0.05) or (per_20_left < 0.05) or ((per_80_left-per_20_left) < 0.05) or ((per_80_right-per_20_right) < 0.05):
                    return False
            np.save(f'{store_path}/{idx:03d}_0_depth.npy',  obs["left_depth_sensor"])
            np.save(f'{store_path}/{idx:03d}_1_depth.npy',  obs["right_depth_sensor"])

        cv2.imwrite(f'{store_path}/{idx:03d}_0.png', obs["left_sensor"][:,:,:3][..., ::-1])
        cv2.imwrite(f'{store_path}/{idx:03d}_1.png', obs["right_sensor"][:,:,:3][..., ::-1])
    return True


def main(display=True, scene_path=None, store_path=None):
    global cv2
    # Only import cv2 if we are doing to display
    if display:
        import cv2 as _cv2

        cv2 = _cv2

        cv2.namedWindow("stereo_pair")

    backend_cfg = habitat_sim.SimulatorConfiguration()
    if scene_path:
        backend_cfg.scene_id = (scene_path)
    else:
        backend_cfg.scene_id = (
            "./scene_datasets/habitat-test-scenes/skokloster-castle.glb"
        )

    # Adjust camera orientation and height each round
    ori = (np.random.rand(3)-0.5)*0.8
    hei = 1.3 + (np.random.rand(1)-0.5)*0.4

    # First, let's create a stereo RGB agent
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    # Give it the uuid of left_sensor, this will also be how we
    # index the observations to retrieve the rendering from this sensor
    left_rgb_sensor.uuid = "left_sensor"
    left_rgb_sensor.resolution = [512, 512]

    left_rgb_sensor.position = hei * habitat_sim.geo.UP + 0.0656 * habitat_sim.geo.LEFT # baseline = 13cm
    left_rgb_sensor.orientation = ori
    left_rgb_sensor.noise_model = "GaussianNoiseModel"
    left_rgb_sensor.noise_model_kwargs = dict(sigma=0.1, intensity_constant=0.1)

    # Same deal with the right sensor
    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.uuid = "right_sensor"
    right_rgb_sensor.resolution = [512, 512]

    right_rgb_sensor.position = hei * habitat_sim.geo.UP + 0.0656 * habitat_sim.geo.RIGHT
    right_rgb_sensor.orientation = ori
    right_rgb_sensor.noise_model = "GaussianNoiseModel"
    right_rgb_sensor.noise_model_kwargs = dict(sigma=0.1, intensity_constant=0.1)

    # Now let's do the exact same thing but for a depth camera stereo pair!
    left_depth_sensor = habitat_sim.CameraSensorSpec()
    left_depth_sensor.uuid = "left_depth_sensor"
    left_depth_sensor.resolution = [512, 512]
    left_depth_sensor.position = hei * habitat_sim.geo.UP + 0.0656 * habitat_sim.geo.LEFT

    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    left_depth_sensor.orientation = ori

    right_depth_sensor = habitat_sim.CameraSensorSpec()
    right_depth_sensor.uuid = "right_depth_sensor"
    right_depth_sensor.resolution = [512, 512]
    right_depth_sensor.position = (
        hei * habitat_sim.geo.UP + 0.0656 * habitat_sim.geo.RIGHT
    )
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    right_depth_sensor.orientation = ori

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, left_depth_sensor, right_depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    sim.seed(110)

    flag = False
    cnt = 0
    flag_last_run = False
    max_run = 300
    while not flag:
        cnt += 1
        flag = _render(sim, display, depth=True, store_path=store_path, last=flag_last_run)
        if cnt == max_run:
            flag_last_run = True
        elif cnt == max_run+1:
            break
    sim.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    args = parser.parse_args()

    # TODO: fill in with the path where you store your {matterport3d, Replica, or HM3D} mesh files 
    PATH = ''
    mp3d_files = sorted(glob.glob(f'{PATH}/*'))
    dest = 'vis/baseline_13cm/mp3d'

    for file in mp3d_files:
        for count in range(0,5): # generate 5 episode, and each is with 50 actions
            name = file.rsplit('/',1)[-1]
            
            # TODO: check if this match your file extension
            glb_path = file + f'/{name}.glb'
            path = f'{dest}/{name}/{count}'
            if not os.path.exists(path):
                os.makedirs(path)

            main(display=args.display, scene_path=glb_path, store_path=path)

