# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import imageio
import json
import kornia
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
from dpt_networks.dpt_depth import DPTDepthModel, DPTDepthModel2


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

        self.use_pose_net = False
        #self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

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

        # Download pretrained weights from DPT (https://github.com/isl-org/DPT) and put them under './weights/'  
        # self.mono_model = DPTDepthModel(
        #     path='./weights/dpt_hybrid-midas-501f0c75.pt',
        #     #path='./weights/dpt_hybrid_nyu-2ce69ec7.pt',
        #     #path='./weights/dpt_large-midas-2f21e586.pt',
        #     backbone="vitb_rn50_384", #DPT-hybrid
        #     #backbone="vitl16_384", # DPT-Large
        #     non_negative=True,
        # )

        # use NYU-finetuned weights
        self.mono_model = DPTDepthModel2(
            path='./weights/dpt_hybrid_nyu-2ce69ec7.pt',
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        self.mono_model.requires_grad=False
        self.mono_model.to(self.device)

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

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate) #optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
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

        # training on VA
        #train_filenames = readlines(fpath.format("VA_all"))
        #val_filenames = readlines(fpath.format("VA_left_all"))

        # training on replica
        train_filenames = readlines(fpath.format("replica_train"))
        val_filenames = readlines(fpath.format("replica_test_sub"))

        # define train/val file list for SimSIN or UniSIN in the under. Please download the data in the project page
        #train_filenames = readlines(fpath.format("all_large_release2")) # readlines(fpath.format("UniSIN_500_list"))
        #val_filenames = readlines(fpath.format("replica_test_sub")

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

        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.depth_criterion = nn.HuberLoss(delta=0.8)
        self.SOFT = nn.Softsign()
        self.ABSSIGN = torch.sign

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

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
        self.cnt = -1

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            self.mode = 'train'
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses_Hab(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
        Pass a minibatch through the network and generate images and losses
        """
        # input images are in [0,1]

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        outputs = self.models["depth"](features)

        # monocular depth cues. Note that it needs to normalize from [0,1] to [-1,-1] for DPT. Also larger denominator needs to be used.
        # comment this out at test time to speed up
        outputs["fromMono"], feature_dpt = self.mono_model((inputs[("color_aug", 0, 0)]-0.5)/0.5 ) 
        outputs["fromMono_dep"] = (1/(outputs["fromMono"]+1e-6)) ##The output range of fromMono is large 250-2500
        #outputs["fromMono_dep"] = outputs["fromMono"] #600.0 * (1/(outputs["fromMono"]+1e-6)) ##The output range of fromMono is large 250-2500

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        if self.mode == 'train':
            losses = self.compute_losses(inputs, outputs)
        elif self.mode == 'val':
            losses={}

        return outputs, losses

    def val(self):
        """
        Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses_Hab(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def compute_losses(self, inputs, outputs, feats=None):
        """
        Combining monodepth2 and distillation losses
        """
        losses = {}
        stereo_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            # only use souce scale for loss
            source_scale = 0

            disp = outputs[("out", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # auto-masking
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            # save both images, and do min all at once below
            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            if combined.shape[1] == 1:
                to_optimize = combined
            else:
                to_optimize, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimize.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            stereo_loss += loss
            losses["loss/{}".format(scale)] = loss

        stereo_loss /= self.num_scales

        losses["loss"] = stereo_loss

        # median alignment for ease of use
        fac = (torch.median(outputs[('depth', 0, 0)]) / torch.median(outputs["fromMono_dep"])).detach()
        target_depth = outputs["fromMono_dep"]*fac

        # spatial gradient
        edge_target = kornia.filters.spatial_gradient(target_depth)
        edge_pred = kornia.filters.spatial_gradient(outputs[('depth', 0, 0)])

        # convert to magnitude map
        edge_target =  torch.sqrt(edge_target[:,:,0,:,:]**2 + edge_target[:,:,1,:,:]**2 + 1e-6)
        edge_target = edge_target[:,:,5:-5,5:-5]
        # thresholding
        bar_target = torch.quantile(edge_target, self.opt.thre)
        pos = edge_target > bar_target
        mask_target = self.ABSSIGN(edge_target - bar_target)[pos]
        mask_target = mask_target.detach()

        # convert prediction to magnitude map 
        edge_pred =  torch.sqrt(edge_pred[:,:,0,:,:]**2 + edge_pred[:,:,1,:,:]**2 + 1e-6)
        edge_pred = F.normalize(edge_pred.view(edge_pred.size(0), -1), dim=1, p=2).view(edge_pred.size())
        edge_pred = edge_pred[:,:,5:-5,5:-5]
        bar_pred = torch.quantile(edge_pred, self.opt.thre).detach()

        # soft sign for differentiable
        mask_pred = self.SOFT(edge_pred - bar_pred)[pos]

        loss_depth_criterion = 0.001 * self.depth_criterion(mask_pred, mask_target)
        losses["loss/pseudo_depth"] = self.compute_ssim_loss(outputs["fromMono_dep"], outputs[('depth', 0, 0)]).mean() + loss_depth_criterion
        losses["loss"] += self.opt.dist_wt * losses["loss/pseudo_depth"]
        print(self.cnt)
        self.cnt += 1
        print(losses["loss"])

        return losses
    
    def compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_ssim_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        """
        return self.ssim(pred, target).mean(1, True)

    def compute_depth_losses_Hab(self, inputs, outputs, losses):
        """
        Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [512, 512], mode="bilinear", align_corners=False), 1e-3, 10)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = torch.logical_and(depth_gt>0.01, depth_gt<=10.0)
        #mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("out", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            depth = output_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # auto-masking
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def eval_save(self):
        """
        save prediction for a minibatch
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

    def eval_save_all(self):
        """
        save prediction for all data on the list
        """
        self.set_eval()
        self.count = 0
        while True:
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                break

            with torch.no_grad():
                inputs[("color_aug", 0, 0)] = inputs[("color_aug", 0, 0)].cuda()
                features = self.models["encoder"](inputs[("color_aug", 0, 0)]) #
                outputs = self.models["depth"](features)
                depth = output_to_depth(outputs[('out', 0)], self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", 0, 0)] = depth
                sz = (640,640)
                store_path = f'results_all/'
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                    os.makedirs(store_path+'/depth')

                img = inputs[('color',0, 0)]
                img = F.interpolate(img, sz, mode='bilinear', align_corners=True)
                img = img.cpu().numpy().squeeze().transpose(0,2,3,1)

                if 'depth_gt' in inputs:
                    depth_gt = inputs['depth_gt']
                    depth_gt = F.interpolate(depth_gt, sz, mode='bilinear', align_corners=True)
                    depth_gt = depth_gt.cpu().numpy().squeeze()
                    mask = depth_gt > 0 

                depth = outputs[('depth', 0, 0)] * self.approx_alignment #approximate alignment for visualization
                depth = F.interpolate(depth, sz, mode='bilinear', align_corners=True)
                depth = depth.cpu().numpy().squeeze()

                batch_size = img.shape[0]
                for idx in range(batch_size):
                    imageio.imwrite(f'{store_path}/{self.count:04d}_img.png', img[idx])
                    write_turbo_depth_metric(f'{store_path}/depth/{self.count:04d}_depth.png', depth, vmax=10.0)
                    self.count += 1
                    if 'depth_gt' in inputs:
                        write_turbo_depth_metric(f'{store_path}/depth/{self.count:04}_depth_gt.png', depth_gt, vmax=10.0)

            del inputs, outputs

    def eval_measure(self):
        """
        eval on either VA or NYUv2
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
                    if self.opt.dataset == 'VA':
                        self.compute_depth_errors_VA(inputs, outputs, losses)
                    elif self.opt.dataset == 'NYUv2':
                        self.compute_depth_errors_NYUv2(inputs, outputs, losses)
                    else:
                        raise NotImplementedError("Do evaluation only on VA or NYUv2")
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
        """
        compute depth errors on VA
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

    def compute_depth_errors_NYUv2(self, inputs, outputs, losses):
        """
        compute depth errors on NYUv2
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(depth_pred, [448, 608], mode="bilinear", align_corners=True)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]

        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10.0)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        if losses is None:
            losses = {}
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """
        print a logging statement to the terminal
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
        """
        write an event to the tensorboard events file
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
                    "out_{}/{}".format(s, j),
                    normalize_image(outputs[("out", s)][j]), self.step)

                # automasking
                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
    
    def log_losses(self, mode, losses):
        """
        write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """
        save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """
        save model weights to disk
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
        """
        load model(s) from disk
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

    def eval_measure_multi(self):
        self.dataset = datasets.VADataset
        fpath = os.path.join(self.opt.data_path,  "{}.txt")
        val_filenames = readlines(fpath.format("UE4_left_freq_5"))
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=8, pin_memory=False, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        self.models["mono_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
        self.models["mono_depth"].to(self.device)

        self.models["pose_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        if self.opt.load_weights_folder is not None:
            self.load_model_multi()


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

        #count = 0
        while True:
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                break

            with torch.no_grad():
                outputs, losses = self.process_batch_multi(inputs)

                if "depth_gt" in inputs:
                    self.self.compute_depth_errors_VA(inputs, outputs, losses)
                    self.abs_mn.update(losses['de/abs_mn'], N)
                    self.abs_rel.update(losses['de/abs_rel'], N)
                    self.sq_rel.update(losses['de/sq_rel'], N)
                    self.rms.update(losses['de/rms'], N)
                    self.log_rms.update(losses['de/log_rms'], N)
                    self.a1.update(losses['da/a1'], N)
                    self.a2.update(losses['da/a2'], N)
                    self.a3.update(losses['da/a3'], N)

        del inputs, outputs, losses

        f = open('test.txt','w')
        #f = open(f'{self.opt.eval_filename}','w')
        all_errors = [self.abs_mn, self.abs_rel, self.sq_rel, self.rms, self.log_rms, self.a1, self.a2, self.a3]
        for comp in all_errors:
            f.write(str(comp))
        f.close()

    def predict_poses(self, inputs):
        """
        Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs

    def process_batch_multi(self, inputs, is_train=False):
        """
        Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        with torch.no_grad():
            pose_pred = self.predict_poses(inputs, None)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        # single frame path

        with torch.no_grad():
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0])
            mono_outputs.update(self.models['mono_depth'](feats))

        self.generate_images_pred(inputs, mono_outputs)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, _, _ = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin)
        outputs.update(self.models["depth"](features))

        self.generate_images_pred_multi(inputs, outputs)
        losses = {}

        return outputs, losses

    def generate_images_pred_multi(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("out", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            depth = output_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

    def load_mono_model(self):
        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model_multi(self):
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

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Can't load Adam - using random")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")