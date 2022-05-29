# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import copy
import imageio
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.backends.cudnn.benchmark = True


from utils import *
from layers import *
import datasets
import networks
from dpt_networks.dpt_depth import DPTDepthModel

SCALE_FAC = 1.0#1.31202#1.2#2#1.2 #2#1.31202 #1.2#2#1.2 #2 720/256 #* 0.82 #1.19912341237#1.0 #2 #1.19912341237 * (720/256) #*0.82 #* 1.19912341237

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # self.mono_model = DPTDepthModel2(
        #     path='./weights/dpt_hybrid_nyu-2ce69ec7.pt',
        #     scale=0.000305,
        #     shift=0.1378,
        #     invert=True,
        #     backbone="vitb_rn50_384",
        #     non_negative=True,
        # )
        # self.mono_model.requires_grad=False
        # self.mono_model.to(self.device)

        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                50,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.SGD(self.parameters_to_train, self.opt.learning_rate) #optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {
                         "SimSIN": datasets.SimSINDataset,
                         "VA": datasets.VADataset,
                         "NYUv2": datasets.NYUv2Dataset,
                         "UniSIN": datasets.UniSINDataset,}
        self.dataset = datasets_dict[self.opt.dataset]

        if self.opt.dataset == 'SimSIN':
            self.approx_factor = 1.0
        elif self.opt.dataset == 'VA':
            self.approx_factor = 2.0
        else:
            self.approx_factor = 1.0


        fpath = os.path.join(self.opt.data_path,  "{}.txt")

        train_filenames = readlines(fpath.format("UE4_all"))
        val_filenames = readlines(fpath.format("UE4_left_all"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_mn", "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        try:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))
        except:
            print("In reference mode! There are {:d} samples\n".format(len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def eval_save(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
            features = self.models["encoder"](inputs[("color_aug", 0, 0)]) #
            outputs = self.models["depth"](features)
            depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, 0)] = depth

            sz = (640,640)
            store_path = 'results/VA'
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            img = inputs[('color_aug',0, 0)]
            img = F.interpolate(img, sz, mode='bilinear', align_corners=True)
            img = img.cpu().numpy().squeeze().transpose(0,2,3,1)

            depth = outputs[('depth', 0, 0)] * self.approx_alignment #approximate alignment for visualization
            depth = F.interpolate(depth, sz, mode='bilinear', align_corners=True)
            depth = depth.cpu().numpy().squeeze()

            bsz = img.shape[0]

            for idx in range(bsz):
                imageio.imwrite(f'{store_path}/{idx:02d}_current.png', img[idx])
                write_turbo_depth_metric(f'{store_path}/{idx:02d}_depth.png', depth[idx], vmax=10.0)

            del inputs, outputs, losses

    def eval_measure(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        self.abs_mn = AverageMeter('abs_mean')
        self.abs_rel = AverageMeter('abs_rel')
        self.sq_rel = AverageMeter('sq_rel')
        self.rms = AverageMeter('rms')
        self.log_rms = AverageMeter('log_rms')
        self.a1 = AverageMeter('a1')
        self.a2 = AverageMeter('a2')
        self.a3 = AverageMeter('a3')
        N = self.opt.batch_size

        local_count = 0
        losses = {}
        while True:
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                if not local_count == 0:
                    break
                else:
                    self.val_iter = iter(self.val_loader)
                    inputs = self.val_iter.next()

            with torch.no_grad():
                inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
                inputs["depth_gt"] = inputs["depth_gt"].cuda()
                features = self.models["encoder"](inputs[("color_aug", 0, 0)])
                outputs = self.models["depth"](features)
                depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, 0)] = depth
                if "depth_gt" in inputs:
                    self.compute_depth_errors_VA(inputs, outputs, losses)
                    self.abs_mn.update(losses['de/abs_mn'], N)
                    self.abs_rel.update(losses['de/abs_rel'], N)
                    self.sq_rel.update(losses['de/sq_rel'], N)
                    self.rms.update(losses['de/rms'], N)
                    self.log_rms.update(losses['de/log_rms'], N)
                    self.a1.update(losses['da/a1'], N)
                    self.a2.update(losses['da/a2'], N)
                    self.a3.update(losses['da/a3'], N)

            local_count += 1

        del inputs, outputs, losses

        idfy = self.opt.load_weights_folder
        f = open(f'evaluation-{idfy}.txt','w')
        all_errors = [self.abs_mn, self.abs_rel, self.sq_rel, self.rms, self.log_rms, self.a1, self.a2, self.a3]
        for comp in all_errors:
            f.write(str(comp))
        f.close()

    def compute_depth_errors_VA(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(depth_pred, [640, 640], mode="bilinear", align_corners=True)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0.01
        mask = torch.logical_and(depth_gt > 0.01, depth_gt<=10.0)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10.0)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        if losses is None:
            losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
    
    def log_losses(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)