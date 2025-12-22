# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import yaml
import os
import time
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from option import Option
import dataset
import utils
import utils.tools as tools
from utils.metrics.eval_results import eval_results
from utils.metrics.tensorboard_logger import tensorboard_logger
from utils.inference.inference_utils import inference
from utils.tools import Recorder


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None, mlflow_manager=None):
        # Init params
        self.settings = settings
        self.recorder = recorder
        self.mlflow_manager = mlflow_manager
        self.model = model.cuda()
        self.use_knn = (not self.settings.use_kpconv) and self.settings.use_knn
        self.boundary_loss_weight = 0.0 if self.settings.use_kpconv else max(0.0, float(getattr(self.settings, 'boundary_loss_weight', 0.0)))
        self.remain_time = tools.RemainTime(self.settings.n_epochs)
        self.iter_steps = {'Train': 0, 'Validation': 0}

        # Init data loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # Init criterion
        self.criterion = self._initCriterion()

        # Init optimizer
        self.optimizer = self._initOptimizer()

        if tools.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = nn.parallel.DistributedDataParallel(
                	self.model, device_ids=[self.settings.gpu],
                    find_unused_parameters=True)

        # Get metrics
        self.metrics = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics.reset()
        self.metrics_3d = None
        self.knn_post = None
        if self.use_knn:
            knn_params = {
                'knn': self.settings.knn_k,
                'search': self.settings.knn_search,
                'sigma': self.settings.knn_sigma,
                'cutoff': self.settings.knn_cutoff,
            }
            self.knn_post = utils.postproc.KNN(
                params=knn_params, nclasses=self.settings.n_classes)
            self.metrics_3d = utils.metrics.IOUEval(
                n_classes=self.settings.n_classes, device=torch.device('cpu'),
                ignore=self.ignore_class, is_distributed=self.settings.distributed)
            self.metrics_3d.reset()

        # Define scheduler
        self.scheduler = utils.optim.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=0.9,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

        # For mixed precision training
        self.fp16_scaler = None
        if self.settings.use_fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()

    def _initOptimizer(self):
        params = self.model.parameters()
        adamw_optimizer = torch.optim.AdamW(params=params,
                                lr=self.settings.lr,
                                weight_decay=0.01)
        return adamw_optimizer

    def _initDataloader(self):
        # NuScenes dataset
        if self.settings.dataset == 'nuScenes':
            print('----Using nuScenes dataset----')
            version = 'v1.0-mini' if self.settings.use_mini_version else 'v1.0-trainval'
            assert self.settings.use_trainval is False

            trainset = dataset.nuScenes.Nuscenes(
                dataroot=self.settings.data_root, version=version, split='train')
            valset = dataset.nuScenes.Nuscenes(
                dataroot=self.settings.data_root, version=version, split='val')

            self.mapped_cls_name = trainset.mapped_cls_name
            self.ignore_class = [0]
            self.cls_weight = np.ones((self.settings.n_classes))
            self.cls_weight[0] = 0
            assert self.settings.test_split is False
            self.data_split = 'test' if self.settings.test_split else 'val'

        # SemanticKitti dataset
        elif self.settings.dataset == 'SemanticKitti':
            data_config_path = 'dataset/semantic_kitti/semantic-kitti.yaml'
            data_config = yaml.safe_load(open(data_config_path, 'r'))

            if self.settings.use_mini_version:
                train_sequences = [0]
            elif self.settings.use_trainval:
                print('Train with the train+val set.')
                train_sequences = data_config['split']['train'] + data_config['split']['valid']
            else:
                train_sequences = data_config['split']['train']

            trainset = dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=train_sequences,
                config_path=data_config_path)

            self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
            self.ignore_class = []
            for cl, _ in enumerate(self.cls_weight):
                if trainset.data_config['learning_ignore'][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info('weight: {}'.format(self.cls_weight))
            self.mapped_cls_name = trainset.mapped_cls_name

            test_sequences = (
                data_config['split']['test'] if self.settings.test_split else
                data_config['split']['valid'])

            valset = dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=test_sequences,
                config_path=data_config_path,
                has_label=(self.settings.test_split is False),
            )

        else:
            raise ValueError(
                'invalid dataset: {}'.format(self.settings.dataset))

        self.train_range_loader = dataset.RangeViewLoader(
            dataset=trainset,
            config=self.settings.config,
            use_kpconv=self.settings.use_kpconv)

        self.val_range_loader = dataset.RangeViewLoader(
            dataset=valset,
            config=self.settings.config,
            is_train=False,
            return_uproj=self.use_knn,
            use_kpconv=self.settings.use_kpconv)

        collate_fn = dataset.custom_collate_kpconv_fn if self.settings.use_kpconv else None
        if tools.is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)

            train_loader = torch.utils.data.DataLoader(
                self.train_range_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.num_workers,
                drop_last=True,
                sampler=train_sampler,
                collate_fn=collate_fn)

            val_loader = torch.utils.data.DataLoader(
                self.val_range_loader,
                batch_size=self.settings.batch_size_val,
                num_workers=self.settings.num_workers,
                drop_last=False,
                sampler=val_sampler,
                collate_fn=collate_fn)

            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_range_loader,
                batch_size=self.settings.batch_size,
                num_workers=self.settings.num_workers,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn)

            val_loader = torch.utils.data.DataLoader(
                self.val_range_loader,
                batch_size=self.settings.batch_size_val,
                num_workers=self.settings.num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn)

            return train_loader, val_loader, None, None

    def _initCriterion(self):
        criterion = {}
        criterion['lovasz'] = utils.optim.Lovasz_softmax(ignore=0)

        if self.settings.dataset == 'SemanticKitti':
            alpha = np.log(1+self.cls_weight)
            alpha = alpha / alpha.max()
        elif self.settings.dataset == 'nuScenes':
            alpha = np.ones((self.settings.n_classes))
        alpha[0] = 0
        if self.recorder is not None:
            self.recorder.logger.info('focal_loss alpha: {}'.format(alpha))

        criterion['focal_loss'] = utils.optim.FocalSoftmaxLoss(
            self.settings.n_classes, gamma=2, alpha=alpha, softmax=False)

        # Set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    def _compute_boundary_mask(self, label: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute binary boundary mask from integer labels (B x H x W).
        Only considers pixels where valid_mask is True; void (label==0) is ignored.
        """
        # Ensure invalid pixels are voided
        label = label * valid_mask.long()
        boundary = torch.zeros_like(label, dtype=torch.bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_label = torch.roll(label, shifts=(dy, dx), dims=(1, 2))
            shifted_valid = torch.roll(valid_mask, shifts=(dy, dx), dims=(1, 2))
            diff = (label != shifted_label) & valid_mask & shifted_valid
            boundary = boundary | diff
        return boundary.float() * valid_mask.float()

    def _compute_pred_boundary_prob(self, prob: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary probability from class probabilities (B x C x H x W).
        Uses neighbor similarity; higher dissimilarity -> higher boundary probability.
        """
        prob_valid = prob * valid_mask.unsqueeze(1)
        sims = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_prob = torch.roll(prob_valid, shifts=(dy, dx), dims=(2, 3))
            sims.append((prob_valid * shifted_prob).sum(dim=1))
        sims = torch.stack(sims, dim=0)
        max_sim, _ = sims.max(dim=0)
        boundary_prob = 1.0 - max_sim
        boundary_prob = boundary_prob * valid_mask
        return boundary_prob.clamp(0.0, 1.0)

    def boundary_loss(self, prob: torch.Tensor, label: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Boundary mIoU-style loss on range image before post-processing.
        - Ignores void/invalid pixels (label == 0 or mask == 0).
        - Returns 1 - boundary IoU.
        """
        valid_mask = (mask > 0) & (label != 0)
        if valid_mask.sum() == 0:
            return torch.zeros([], device=prob.device, dtype=prob.dtype)

        target_boundary = self._compute_boundary_mask(label, valid_mask)
        pred_boundary = self._compute_pred_boundary_prob(prob, valid_mask)

        intersection = (pred_boundary * target_boundary).sum(dim=(1, 2))
        union = (pred_boundary + target_boundary - pred_boundary * target_boundary).sum(dim=(1, 2)) + 1e-6
        iou = intersection / union
        return 1.0 - iou.mean()

    def compute_losses(self, output, output_softmax, label, mask):
        loss_lovasz = self.criterion['lovasz'](output_softmax, label)
        loss_focal = self.criterion['focal_loss'](output_softmax, label, mask=mask)
        total_loss = loss_focal + loss_lovasz

        loss_boundary = torch.zeros([], device=output.device, dtype=output.dtype)
        if self.boundary_loss_weight > 0:
            loss_boundary = self.boundary_loss(output_softmax, label, mask)
            total_loss = total_loss + self.boundary_loss_weight * loss_boundary

        return total_loss, loss_lovasz, loss_focal, loss_boundary


    def run(self, epoch, mode='Train', print_results=False, save_results_path=None):
        if self.settings.use_kpconv:
            # Training and validation when using the KPConv layer
            return self.run_with_kpconv(
                epoch=epoch, mode=mode,
                print_results=print_results,
                save_results_path=save_results_path)
        else:
            # Training and validation when not using the KPConv layer
            return self.run_without_kpconv(
                epoch=epoch, mode=mode,
                print_results=print_results,
                save_results_path=save_results_path)           

    # Method for training when using the KPConv layer
    def run_without_kpconv(self, epoch, mode='Train', print_results=False, save_results_path=None):
        if mode == 'Train':
            dataloader = self.train_loader
            self.model.train()
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
        elif mode == 'Validation':
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module

        # Init metrics
        loss_meter = tools.AverageMeter()
        self.metrics.reset()
        if mode == 'Validation' and self.use_knn:
            self.metrics_3d.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        log_frequency = max(1, self.settings.log_frequency)

        for i, batch in enumerate(dataloader):
            t_process_start = time.time()
            current_lr = None
            if mode == 'Validation' and self.use_knn:
                (input_feature, input_label, input_mask, proj_depth,
                 uproj_x_idx, uproj_y_idx, uproj_depth, sem_label) = batch
            else:
                input_feature, input_label, input_mask = batch

            # Feature: range, x, y, z, intensity
            input_feature = input_feature.cuda() # shape: B x 5 x H x W

            input_label = input_label.cuda().long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.cuda() * input_label.ge(1).float()

            # Forward propagation
            if mode == 'Train':
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output = self.model(input_feature)
                    output_softmax = F.softmax(output, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary = self.compute_losses(
                        output, output_softmax, input_label, input_mask)

                # Backward
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()

                # Update lr after backward (required by pytorch)
                self.scheduler.step()
            with torch.no_grad():
                if mode == 'Validation':
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    im_meta = dict(flip=False)
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        lidar_pred = inference(
                            model_without_ddp.rangevit,
                            [input_feature],
                            [im_meta],
                            ori_shape=input_feature.shape[2:4],
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            batch_size=input_feature.shape[0],
                            use_kpconv=False,
                            use_sliding_window=self.settings.use_sliding_window)

                    output = lidar_pred.unsqueeze(0) # [C, H, W] ==> [1, C, H, W]
                    output_softmax = F.softmax(output, dim=1)
                    if self.use_knn:
                        proj_depth = proj_depth[0].cuda()
                        uproj_x_idx = uproj_x_idx[0].cuda()
                        uproj_y_idx = uproj_y_idx[0].cuda()
                        uproj_depth = uproj_depth[0].cuda()
                        sem_label = sem_label[0].cuda()
                        pred_argmax = output_softmax[0].argmax(dim=0)
                        unproj_argmax = self.knn_post(
                            proj_depth, uproj_depth, pred_argmax, uproj_x_idx, uproj_y_idx)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary = self.compute_losses(
                        output, output_softmax, input_label, input_mask)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                self.metrics.addBatch(argmax, input_label) # 2D predictions
                if mode == 'Validation' and self.use_knn:
                    self.metrics_3d.addBatch(unproj_argmax, sem_label) # 3D predictions

            loss_meter.update(loss.item(), input_feature.size(0))

            with torch.no_grad():
                mean_iou_tensor, _, mean_acc_tensor, _ = self.metrics.getIoUnAcc()
                mean_recall_tensor, _ = self.metrics.getRecall()
            mean_iou_running = float(mean_iou_tensor)
            mean_acc_running = float(mean_acc_tensor)
            mean_recall_running = float(mean_recall_tensor)

            should_log = (i % log_frequency == 0) or (i == total_iter - 1)

            if should_log and self.mlflow_manager is not None:
                step_id = self.iter_steps[mode]
                self.iter_steps[mode] += 1
                mlflow_metrics = {
                    f'{mode.lower()}_loss': loss.item(),
                    f'{mode.lower()}_mean_iou': mean_iou_running,
                    f'{mode.lower()}_mean_acc': mean_acc_running,
                    f'{mode.lower()}_mean_recall': mean_recall_running,
                }
                if mode == 'Train':
                    mlflow_metrics[f'{mode.lower()}_lr'] = current_lr
                self.mlflow_manager.log_metrics(mlflow_metrics, step=step_id)

            # Save predictions for SemanticKITTI test/val when KNN unprojection is used (non-KPConv path).
            if (mode == 'Validation' and save_results_path is not None and self.use_knn):
                pred_np = unproj_argmax.cpu().numpy().reshape(-1).astype(np.int32)
                sk_dataset = self.val_range_loader.dataset
                pred_np_origin = sk_dataset.class_map_lut_inv[pred_np]
                seq_id, frame_id = sk_dataset.parsePathInfoByIndex(i)
                pred_path = os.path.join(save_results_path, 'sequences', seq_id, 'predictions')
                if not os.path.isdir(pred_path):
                    os.makedirs(pred_path)
                pred_result_path = os.path.join(pred_path, f'{frame_id}.label')
                pred_np_origin.tofile(pred_result_path)

            # Timer logger
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                ))
            t_start = time.time()

            # Logging
            if should_log:
                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} '.format(
                        current_lr, loss.item(), mean_acc_running, mean_iou_running)
                    log_str += 'RT {}'.format(remain_time)
                    self.recorder.logger.info(log_str)

        with torch.no_grad():
            mean_acc, class_acc = self.metrics.getAcc()
            mean_recall, class_recall = self.metrics.getRecall()
            mean_iou, class_iou = self.metrics.getIoU()

            metrics_dict = {
                'mean_acc': mean_acc,
                'class_acc': class_acc,
                'mean_recall': mean_recall,
                'class_recall': class_recall,
                'mean_iou': mean_iou,
                'class_iou': class_iou,
                'conf_matrix': self.metrics.conf_matrix.clone().cpu(),
            }
            metrics_dict_3d = None
            if mode == 'Validation' and self.use_knn:
                mean_acc_3d, class_acc_3d = self.metrics_3d.getAcc()
                mean_recall_3d, class_recall_3d = self.metrics_3d.getRecall()
                mean_iou_3d, class_iou_3d = self.metrics_3d.getIoU()
                metrics_dict_3d = {
                    'mean_acc': mean_acc_3d,
                    'class_acc': class_acc_3d,
                    'mean_recall': mean_recall_3d,
                    'class_recall': class_recall_3d,
                    'mean_iou': mean_iou_3d,
                    'class_iou': class_iou_3d,
                    'conf_matrix': self.metrics_3d.conf_matrix.clone().cpu(),
                }

        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
            }

        epoch_lr = self.optimizer.param_groups[0]['lr']

        # Print results
        if self.recorder is not None:
            # Print train pixel-wise evaluation results
            if mode == 'Train':
                if (epoch % self.settings.train_result_frequency == 0) or (epoch == self.settings.n_epochs-1):
                    eval_results(pixel_or_point='Pixel',
                                 settings=self.settings,
                                 recorder=self.recorder,
                                 metrics_dict=metrics_dict,
                                 dataloader=self.train_range_loader,
                                 print_data_distribution=True)

            # Print validation pixel-wise evaluation results
            if mode == 'Validation' and (print_results or epoch == self.settings.n_epochs-1):
                eval_results(pixel_or_point='Pixel',
                             settings=self.settings,
                             recorder=self.recorder,
                             metrics_dict=metrics_dict,
                             dataloader=self.val_range_loader,
                             print_data_distribution=True)
                if self.use_knn and metrics_dict_3d is not None:
                    eval_results(pixel_or_point='Point',
                                 settings=self.settings,
                                 recorder=self.recorder,
                                 metrics_dict=metrics_dict_3d,
                                 dataloader=self.val_range_loader,
                                 print_data_distribution=True)

            # Tensorboard logger
            tensorboard_logger(epoch=epoch,
                               mode=mode,
                               recorder=self.recorder,
                               metrics_dict=metrics_dict,
                               loss_dict=loss_dict,
                               lr=epoch_lr,
                               mapped_cls_name=self.mapped_cls_name)

            # Results at the end of the epoch
            log_str = '>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}'.format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            self.recorder.logger.info(log_str)

        if self.mlflow_manager is not None:
            mlflow_metrics = {
                f'{mode.lower()}_epoch_loss': loss_meter.avg,
                f'{mode.lower()}_epoch_acc': mean_acc.item(),
                f'{mode.lower()}_epoch_iou': mean_iou.item(),
                f'{mode.lower()}_epoch_recall': mean_recall.item(),
            }
            if (mode == 'Train') and (epoch_lr is not None):
                mlflow_metrics[f'{mode.lower()}_epoch_lr'] = epoch_lr
            self.mlflow_manager.log_metrics(mlflow_metrics, step=self.iter_steps[mode])


        result_metrics = {
            'Acc': mean_acc.item(),
            'IOU': mean_iou.item(),
            'Recall': mean_recall.item()
        }

        return result_metrics

    # Method for training and validation when using the KPConv layer
    def run_with_kpconv(self, epoch, mode='Train', print_results=False, save_results_path=None):
        if mode == 'Train':
            dataloader = self.train_loader
            self.model.train()
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
        elif mode == 'Validation':
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        track_remain_time_1epoch = tools.RemainTime(1)

        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module

        # Init metrics
        loss_meter = tools.AverageMeter()
        self.metrics.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        log_frequency = max(1, self.settings.log_frequency)

        for i, batch_dict in enumerate(dataloader):
            t_process_start = time.time()
            current_lr = None

            # 2D inputs
            input_feature = batch_dict['input2d'].cuda(non_blocking=True)
            assert self.settings.in_channels == 5

            # 3D inputs
            py = batch_dict['py'].cuda(non_blocking=True)
            px = batch_dict['px'].cuda(non_blocking=True)
            pxyz = batch_dict['points_xyz'].cuda(non_blocking=True)
            knns = batch_dict['knns'].cuda(non_blocking=True)
            labels3d = batch_dict['labels'].cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
            labels3d = labels3d * labels3d.ge(1).long()
            mask_3d = labels3d.ge(1).float()
            num_points = batch_dict['num_points']

            # Forward propagation
            if mode == 'Train':
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output3d = self.model(input_feature, px, py, pxyz, knns, num_points)

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

                # Backward
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()

                # Update lr after backward (required by pytorch)
                self.scheduler.step()
            with torch.no_grad():
                if mode == 'Validation':
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    im_meta = dict(flip=False)
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        output_features2d = inference(
                            model_without_ddp.rangevit,
                            [input_feature],
                            [im_meta],
                            ori_shape=input_feature.shape[2:4],
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            batch_size=input_feature.shape[0],
                            use_kpconv=True,
                            use_sliding_window=self.settings.use_sliding_window)

                        output_features2d = output_features2d.unsqueeze(0) # [C, H, W] ==> [1, C, H, W]

                        # Apply KPConv layer
                        output3d = model_without_ddp.rangevit.kpclassifier(
                            output_features2d, px, py, pxyz, knns, num_points)

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax3d = output3d.argmax(dim=1)
                self.metrics.addBatch(argmax3d, labels3d) # 3D predictions

            loss_meter.update(loss.item(), input_feature.size(0))

            with torch.no_grad():
                mean_iou_tensor, _, mean_acc_tensor, _ = self.metrics.getIoUnAcc()
                mean_recall_tensor, _ = self.metrics.getRecall()
            mean_iou_running = float(mean_iou_tensor)
            mean_acc_running = float(mean_acc_tensor)
            mean_recall_running = float(mean_recall_tensor)

            should_log = (i % log_frequency == 0) or (i == total_iter - 1)

            if should_log and self.mlflow_manager is not None:
                step_id = self.iter_steps[mode]
                self.iter_steps[mode] += 1
                mlflow_metrics = {
                    f'{mode.lower()}_loss': loss.item(),
                    f'{mode.lower()}_mean_iou': mean_iou_running,
                    f'{mode.lower()}_mean_acc': mean_acc_running,
                    f'{mode.lower()}_mean_recall': mean_recall_running,
                }
                if mode == 'Train':
                    mlflow_metrics[f'{mode.lower()}_lr'] = current_lr
                self.mlflow_manager.log_metrics(mlflow_metrics, step=step_id)

            # Save the predictions
            if (mode == 'Validation' and save_results_path is not None):
                pred_np = argmax3d.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)
                index = batch_dict['index']
                assert index.shape[0] == 1
                index = index.item()
                if self.settings.dataset == 'nuScenes':
                    pred_path = os.path.join(save_results_path, 'lidarseg', self.data_split)
                    nu_dataset = self.val_loader.dataset.dataset
                    lidar_token = nu_dataset.token_list[index]
                    if not os.path.isdir(pred_path):
                        os.makedirs(pred_path)
                    pred_result_path = os.path.join(pred_path, '{}_lidarseg.bin'.format(lidar_token))
                    pred_np.tofile(pred_result_path)

                elif self.settings.dataset == 'SemanticKitti':
                    sk_dataset = self.val_loader.dataset.dataset
                    pred_np_origin = sk_dataset.class_map_lut_inv[pred_np]
                    seq_id, frame_id = sk_dataset.parsePathInfoByIndex(index)
                    pred_path = os.path.join(save_results_path, 'sequences', seq_id, 'predictions')
                    if not os.path.isdir(pred_path):
                        os.makedirs(pred_path)
                    pred_result_path = os.path.join(pred_path, '{}.label'.format(frame_id))
                    pred_np_origin.tofile(pred_result_path)

            # Timer logger
            t_process_end = time.time()
            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode))

            track_remain_time_1epoch.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time_1epoch = datetime.timedelta(
                seconds=track_remain_time_1epoch.getRemainTime(
                    epoch=0, iters=i, total_iter=total_iter, mode=mode))

            t_start = time.time()

            # Logging
            if should_log:
                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} '.format(
                        current_lr, loss.item(), mean_acc_running, mean_iou_running)
                    log_str += 'RT {} '.format(remain_time)
                    log_str += 'RT PER EPOCH {}'.format(remain_time_1epoch)
                    self.recorder.logger.info(log_str)

        with torch.no_grad():
            mean_acc, class_acc = self.metrics.getAcc()
            mean_recall, class_recall = self.metrics.getRecall()
            mean_iou, class_iou = self.metrics.getIoU()

            metrics_dict = {
                'mean_acc': mean_acc,
                'class_acc': class_acc,
                'mean_recall': mean_recall,
                'class_recall': class_recall,
                'mean_iou': mean_iou,
                'class_iou': class_iou,
                'conf_matrix': self.metrics.conf_matrix.clone().cpu(),
            }

        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
            }

        epoch_lr = self.optimizer.param_groups[0]['lr']

        # Print results
        if self.recorder is not None:
            # Print train point-wise results
            if mode == 'Train':
                if (epoch % self.settings.train_result_frequency == 0) or (epoch == self.settings.n_epochs-1):
                    eval_results(pixel_or_point='Point',
                                 settings=self.settings,
                                 recorder=self.recorder,
                                 metrics_dict=metrics_dict,
                                 dataloader=self.train_range_loader,
                                 print_data_distribution=True)

            # Print validation point-wise results
            if mode == 'Validation' and (print_results or epoch == self.settings.n_epochs-1):
                eval_results(pixel_or_point='Point',
                             settings=self.settings,
                             recorder=self.recorder,
                             metrics_dict=metrics_dict,
                             dataloader=self.val_range_loader,
                             print_data_distribution=True)

            # Tensorboard logger
            tensorboard_logger(epoch=epoch,
                               mode=mode,
                               recorder=self.recorder,
                               metrics_dict=metrics_dict,
                               loss_dict=loss_dict,
                               lr=epoch_lr,
                               mapped_cls_name=self.mapped_cls_name)

            # Results at the end of the epoch
            log_str = '>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}'.format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            self.recorder.logger.info(log_str)

        if self.mlflow_manager is not None:
            mlflow_metrics = {
                f'{mode.lower()}_epoch_loss': loss_meter.avg,
                f'{mode.lower()}_epoch_acc': mean_acc.item(),
                f'{mode.lower()}_epoch_iou': mean_iou.item(),
                f'{mode.lower()}_epoch_recall': mean_recall.item(),
            }
            if (mode == 'Train') and (epoch_lr is not None):
                mlflow_metrics[f'{mode.lower()}_epoch_lr'] = epoch_lr
            self.mlflow_manager.log_metrics(mlflow_metrics, step=self.iter_steps[mode])


        result_metrics = {
            'Acc': mean_acc.item(),
            'IOU': mean_iou.item(),
            'Recall': mean_recall.item()
        }

        return result_metrics
