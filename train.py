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
from typing import Callable, Dict, Optional

from option import Option
import dataset
import utils
import utils.tools as tools
from utils.metrics.eval_results import eval_results
from utils.metrics.tensorboard_logger import tensorboard_logger
from utils.inference.inference_utils import inference
from utils.tools import Recorder


class Trainer(object):
    def __init__(
        self,
        settings: Option,
        model: nn.Module,
        recorder=None,
        mlflow_step_logger: Optional[Callable[[str, int, int, Dict[str, float]], None]] = None,
    ):
        # Init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = tools.RemainTime(self.settings.n_epochs)
        self.mlflow_step_logger = mlflow_step_logger

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
                    find_unused_parameters=False)

        # Get metrics
        self.metrics = utils.metrics.IOUEval(
            n_classes=self.settings.n_classes, device=torch.device('cpu'),
            ignore=self.ignore_class, is_distributed=self.settings.distributed)
        self.metrics.reset()
        self.point_metrics = None
        self.knn_post = None
        self.need_point_eval = (not self.settings.use_kpconv)
        self.use_extra_2d_losses = (not self.settings.use_kpconv)
        loss_weights_cfg = getattr(self.settings, 'loss_weights', {}) or {}
        def get_weight(name, default=1.0):
            value = loss_weights_cfg.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)
        self.loss_weight_focal = get_weight('focal', 1.0)
        self.loss_weight_lovasz = get_weight('lovasz', 1.0)
        self.loss_weight_dice = get_weight('dice', 1.0)
        self.loss_weight_boundary = get_weight('boundary', 0.1)
        self.loss_weight_aux = get_weight('aux', 0.4)
        if self.use_extra_2d_losses:
            boundary_kernel = torch.tensor(
                [[0., 1., 0.],
                 [1., -4., 1.],
                 [0., 1., 0.]],
                dtype=torch.float32).view(1, 1, 3, 3)
            self.boundary_kernel = boundary_kernel
        else:
            self.boundary_kernel = None
        if self.need_point_eval:
            self.point_metrics = utils.metrics.IOUEval(
                n_classes=self.settings.n_classes,
                device=torch.device('cpu'),
                ignore=self.ignore_class,
                is_distributed=self.settings.distributed)
            self.point_metrics.reset()

        self.use_knn_post = self.need_point_eval and self.settings.use_knn
        if self.use_knn_post:
            knn_params = {
                'knn': self.settings.knn_k,
                'search': self.settings.knn_search,
                'sigma': self.settings.knn_sigma,
                'cutoff': self.settings.knn_cutoff,
            }
            self.knn_post = utils.postproc.KNN(
                params=knn_params,
                nclasses=self.settings.n_classes)

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

        val_return_uproj = (not self.settings.use_kpconv)
        self.val_range_loader = dataset.RangeViewLoader(
            dataset=valset,
            config=self.settings.config,
            is_train=False,
            return_uproj=val_return_uproj,
            use_kpconv=self.settings.use_kpconv)
        self.val_loader_returns_uproj = val_return_uproj

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

    def compute_losses(self, output, output_softmax, label, mask):
        loss_lovasz = self.criterion['lovasz'](output_softmax, label)
        loss_focal = self.criterion['focal_loss'](output_softmax, label, mask=mask)
        total_loss = (
            self.loss_weight_focal * loss_focal +
            self.loss_weight_lovasz * loss_lovasz)
        loss_dice = None
        loss_boundary = None
        if self.use_extra_2d_losses:
            loss_dice = self.dice_loss(output_softmax, label, mask)
            loss_boundary = self.boundary_loss(output_softmax, label, mask)
            total_loss = total_loss + self.loss_weight_dice * loss_dice + \
                self.loss_weight_boundary * loss_boundary
        return total_loss, loss_lovasz, loss_focal, loss_dice, loss_boundary

    def dice_loss(self, probs, label, mask, eps=1e-6):
        num_classes = self.settings.n_classes
        one_hot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2).float()
        mask = mask.unsqueeze(1)
        probs = probs * mask
        one_hot = one_hot * mask
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        if dice.shape[0] > 1:
            dice = dice[1:]
        if dice.numel() == 0:
            return probs.new_tensor(0.0)
        dice_loss = 1.0 - dice.mean()
        return dice_loss

    def boundary_loss(self, probs, label, mask):
        if self.boundary_kernel is None:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        num_classes = self.settings.n_classes
        kernel = self.boundary_kernel.to(probs.device)
        kernel = kernel.repeat(num_classes, 1, 1, 1)
        mask = mask.unsqueeze(1)
        probs = probs * mask
        one_hot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * mask
        pred_boundary = torch.abs(F.conv2d(
            probs, kernel, padding=1, groups=num_classes))
        gt_boundary = torch.abs(F.conv2d(
            one_hot, kernel, padding=1, groups=num_classes))
        return F.l1_loss(pred_boundary, gt_boundary)


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
        if mode != 'Train' and self.point_metrics is not None:
            self.point_metrics.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        use_uproj_inputs = (mode != 'Train') and getattr(self, 'val_loader_returns_uproj', False)
        loss_aux = torch.tensor(0.0)

        for i, batch in enumerate(dataloader):
            # [TESTING] Early exit option for testing (limit iterations per epoch)
            if self.settings.max_iters_per_epoch is not None and i >= self.settings.max_iters_per_epoch:
                print(f'[TESTING PIPELIONE] Reached max_iters_per_epoch={self.settings.max_iters_per_epoch}, stopping epoch early')
                break

            t_process_start = time.time()

            if use_uproj_inputs:
                (input_feature, input_label, input_mask, proj_depth,
                 uproj_x_idx, uproj_y_idx, uproj_depth, sem_label, sample_idx) = batch
                proj_depth = proj_depth.cuda(non_blocking=True)[0]
                uproj_x_idx = uproj_x_idx.cuda(non_blocking=True)[0]
                uproj_y_idx = uproj_y_idx.cuda(non_blocking=True)[0]
                uproj_depth = uproj_depth.cuda(non_blocking=True)[0]
                sem_label = sem_label[0]
                if self.settings.test_split:
                    sem_label = None
            else:
                input_feature, input_label, input_mask = batch
                proj_depth = uproj_x_idx = uproj_y_idx = uproj_depth = sem_label = sample_idx = None

            # Feature: range, x, y, z, intensity
            input_feature = input_feature.cuda() # shape: B x 5 x H x W

            input_label = input_label.cuda().long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.cuda() * input_label.ge(1).float()

            loss_aux = None
            # Forward propagation
            if mode == 'Train':
                with torch.amp.autocast('cuda', enabled=self.fp16_scaler is not None):
                    model_output = self.model(input_feature)
                    aux_outputs = None
                    if isinstance(model_output, (tuple, list)):
                        output = model_output[0]
                        aux_outputs = model_output[1]
                    else:
                        output = model_output

                    output_softmax = F.softmax(output, dim=1)

                    # Loss calculation
                    (total_loss,
                     loss_lovasz,
                     loss_focal,
                     loss_dice,
                     loss_boundary) = self.compute_losses(
                        output, output_softmax, input_label, input_mask)
                    loss_aux = None
                    if aux_outputs is not None and len(aux_outputs) > 0:
                        aux_losses = []
                        for aux_logit in aux_outputs:
                            aux_softmax = F.softmax(aux_logit, dim=1)
                            aux_total_loss, _, _, _, _ = self.compute_losses(
                                aux_logit, aux_softmax, input_label, input_mask)
                            aux_losses.append(aux_total_loss)
                        if len(aux_losses) > 0:
                            loss_aux = torch.stack(aux_losses).mean()
                            total_loss = total_loss + self.loss_weight_aux * loss_aux
                    if loss_aux is None:
                        loss_aux = torch.zeros(1, device=total_loss.device, dtype=total_loss.dtype)

                # Backward
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                    # Update lr after optimizer step (required by pytorch)
                    self.scheduler.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    # Update lr after optimizer step (required by pytorch)
                    self.scheduler.step()
                    self.fp16_scaler.update()
            else:
                with torch.no_grad():
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    im_meta = dict(flip=False)
                    with torch.amp.autocast('cuda', enabled=self.fp16_scaler is not None):
                        lidar_pred = inference(
                            model_without_ddp.rangevit,
                            [input_feature],
                            [im_meta],
                            ori_shape=input_feature.shape[2:4],
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            batch_size=input_feature.shape[0],
                            use_kpconv=False)

                    output = lidar_pred.unsqueeze(0) # [C, H, W] ==> [1, C, H, W]
                    output_softmax = F.softmax(output, dim=1)

                    # Loss calculation
                    (total_loss,
                     loss_lovasz,
                     loss_focal,
                     loss_dice,
                     loss_boundary) = self.compute_losses(
                        output, output_softmax, input_label, input_mask)
                    loss_aux = torch.zeros(1, device=total_loss.device, dtype=total_loss.dtype)

                    if use_uproj_inputs:
                        pred_argmax = output.argmax(dim=1)[0]
                        if self.use_knn_post and self.knn_post is not None:
                            unproj_argmax = self.knn_post(
                                proj_depth,
                                uproj_depth,
                                pred_argmax,
                                uproj_x_idx,
                                uproj_y_idx)
                        else:
                            unproj_argmax = pred_argmax[uproj_y_idx, uproj_x_idx]

                        if self.point_metrics is not None and (sem_label is not None) and (not self.settings.test_split):
                            self.point_metrics.addBatch(
                                unproj_argmax.cpu().numpy(),
                                sem_label.numpy())

                        if (mode == 'Validation' and save_results_path is not None):
                            pred_np = unproj_argmax.cpu().numpy().reshape((-1,)).astype(np.int32)
                            sample_idx_int = sample_idx.item() if torch.is_tensor(sample_idx) else int(sample_idx)
                            if self.settings.dataset == 'NuScenes':
                                pred_path = os.path.join(save_results_path, 'lidarseg', self.data_split)
                                nu_dataset = self.val_loader.dataset.dataset
                                lidar_token = nu_dataset.token_list[sample_idx_int]
                                if not os.path.isdir(pred_path):
                                    os.makedirs(pred_path)
                                pred_result_path = os.path.join(pred_path, '{}_lidarseg.bin'.format(lidar_token))
                                pred_np.tofile(pred_result_path)
                            elif self.settings.dataset == 'SemanticKitti':
                                sk_dataset = self.val_loader.dataset.dataset
                                pred_np_origin = sk_dataset.class_map_lut_inv[pred_np]
                                seq_id, frame_id = sk_dataset.parsePathInfoByIndex(sample_idx_int)
                                pred_path = os.path.join(save_results_path, 'sequences', seq_id, 'predictions')
                                if not os.path.isdir(pred_path):
                                    os.makedirs(pred_path)
                                pred_result_path = os.path.join(pred_path, '{}.label'.format(frame_id))
                                pred_np_origin.tofile(pred_result_path)


            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                self.metrics.addBatch(argmax, input_label) # 2D predictions

            loss_meter.update(loss.item(), input_feature.size(0))

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
            if (i % self.settings.log_frequency == 0) or (i == total_iter-1):
                with torch.no_grad():
                    mean_iou, _, mean_acc, _ = self.metrics.getIoUnAcc()
                lr_value = None
                for g in self.optimizer.param_groups:
                    lr_value = g['lr']
                    break

                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} '.format(
                        lr_value, loss.item(), mean_acc.item(), mean_iou.item())
                    log_str += 'RT {}'.format(remain_time)
                    self.recorder.logger.info(log_str)

                if self.mlflow_step_logger is not None and not self.settings.test_split:
                    step_index = epoch * total_iter + i
                    metrics_payload = {
                        'loss': float(loss.item()),
                        'acc': float(mean_acc.item()),
                        'iou': float(mean_iou.item()),
                        'loss_focal': float(loss_focal.item()),
                        'loss_lovasz': float(loss_lovasz.item()),
                        'loss_aux': float(loss_aux.item()) if loss_aux is not None else 0.0,
                        'loss_dice': float(loss_dice.item()) if loss_dice is not None else 0.0,
                        'loss_boundary': float(loss_boundary.item()) if loss_boundary is not None else 0.0,
                    }
                    if mode == 'Train' and lr_value is not None:
                        metrics_payload['lr'] = float(lr_value)
                    self.mlflow_step_logger(mode, epoch, step_index, metrics_payload)

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
            point_metrics_dict = None
            if use_uproj_inputs and (self.point_metrics is not None) and (not self.settings.test_split):
                pt_mean_acc, pt_class_acc = self.point_metrics.getAcc()
                pt_mean_recall, pt_class_recall = self.point_metrics.getRecall()
                pt_mean_iou, pt_class_iou = self.point_metrics.getIoU()
                point_metrics_dict = {
                    'mean_acc': pt_mean_acc,
                    'class_acc': pt_class_acc,
                    'mean_recall': pt_mean_recall,
                    'class_recall': pt_class_recall,
                    'mean_iou': pt_mean_iou,
                    'class_iou': pt_class_iou,
                    'conf_matrix': self.point_metrics.conf_matrix.clone().cpu(),
                }

        zero_like = loss_focal.new_tensor(0.0)
        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
                'loss_aux': loss_aux,
                'loss_dice': loss_dice if loss_dice is not None else zero_like,
                'loss_boundary': loss_boundary if loss_boundary is not None else zero_like,
            }

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
                if point_metrics_dict is not None:
                    eval_results(pixel_or_point='Point',
                                 settings=self.settings,
                                 recorder=self.recorder,
                                 metrics_dict=point_metrics_dict,
                                 dataloader=self.val_range_loader,
                                 print_data_distribution=True)

            # Tensorboard logger (skip for test split - no labels available)
            if not self.settings.test_split:
                tensorboard_logger(epoch=epoch,
                                   mode=mode,
                                   recorder=self.recorder,
                                   metrics_dict=metrics_dict,
                                   loss_dict=loss_dict,
                                   lr=lr_value,
                                   mapped_cls_name=self.mapped_cls_name)

                # Results at the end of the epoch
                log_str = '>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}'.format(
                    mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
                self.recorder.logger.info(log_str)
            else:
                self.recorder.logger.info(f'>>> {mode} on test split completed - predictions saved')


        # Return metrics only if labels are available
        if not self.settings.test_split:
            result_metrics = {
                'Acc': mean_acc.item(),
                'IOU': mean_iou.item(),
                'Recall': mean_recall.item()
            }
            if point_metrics_dict is not None:
                result_metrics.update({
                    'PointAcc': point_metrics_dict['mean_acc'].item(),
                    'PointIOU': point_metrics_dict['mean_iou'].item(),
                    'PointRecall': point_metrics_dict['mean_recall'].item(),
                })
            return result_metrics
        else:
            # Test split has no labels, return None or dummy metrics
            return None

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

        for i, batch_dict in enumerate(dataloader):
            # Early exit for testing (limit iterations per epoch)
            if self.settings.max_iters_per_epoch is not None and i >= self.settings.max_iters_per_epoch:
                print(f'[TESTING PIPELINE] Reached max_iters_per_epoch={self.settings.max_iters_per_epoch}, stopping epoch early')
                break

            t_process_start = time.time()

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
                with torch.amp.autocast('cuda', enabled=self.fp16_scaler is not None):
                    output3d = self.model(input_feature, px, py, pxyz, knns, num_points)

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    (total_loss,
                     loss_lovasz,
                     loss_focal,
                     loss_dice,
                     loss_boundary) = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

                # Backward
                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    self.optimizer.step()
                    # Update lr after optimizer step (required by pytorch)
                    self.scheduler.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.step(self.optimizer)
                    # Update lr after optimizer step (required by pytorch)
                    self.scheduler.step()
                    self.fp16_scaler.update()
            else:
                with torch.no_grad():
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    im_meta = dict(flip=False)
                    with torch.amp.autocast('cuda', enabled=self.fp16_scaler is not None):
                        output_features2d = inference(
                            model_without_ddp.rangevit,
                            [input_feature],
                            [im_meta],
                            ori_shape=input_feature.shape[2:4],
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            batch_size=input_feature.shape[0],
                            use_kpconv=True)

                        output_features2d = output_features2d.unsqueeze(0) # [C, H, W] ==> [1, C, H, W]

                        # Apply KPConv layer
                        output3d = model_without_ddp.rangevit.kpclassifier(
                            output_features2d, px, py, pxyz, knns, num_points)

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    (total_loss,
                     loss_lovasz,
                     loss_focal,
                     loss_dice,
                     loss_boundary) = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax3d = output3d.argmax(dim=1)
                self.metrics.addBatch(argmax3d, labels3d) # 3D predictions

            loss_meter.update(loss.item(), input_feature.size(0))

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
            if (i % self.settings.log_frequency == 0) or (i == total_iter-1):
                with torch.no_grad():
                    mean_iou, _, mean_acc, _ = self.metrics.getIoUnAcc()

                lr_value = None
                for g in self.optimizer.param_groups:
                    lr_value = g['lr']
                    break

                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} '.format(
                        lr_value, loss.item(), mean_acc.item(), mean_iou.item())
                    log_str += 'RT {} '.format(remain_time)
                    log_str += 'RT PER EPOCH {}'.format(remain_time_1epoch)
                    self.recorder.logger.info(log_str)

                if self.mlflow_step_logger is not None and not self.settings.test_split:
                    step_index = epoch * total_iter + i
                    metrics_payload = {
                        'loss': float(loss.item()),
                        'acc': float(mean_acc.item()),
                        'iou': float(mean_iou.item()),
                        'loss_focal': float(loss_focal.item()),
                        'loss_lovasz': float(loss_lovasz.item()),
                        'loss_aux': 0.0,
                        'loss_dice': float(loss_dice.item()) if loss_dice is not None else 0.0,
                        'loss_boundary': float(loss_boundary.item()) if loss_boundary is not None else 0.0,
                    }
                    if mode == 'Train' and lr_value is not None:
                        metrics_payload['lr'] = float(lr_value)
                    self.mlflow_step_logger(mode, epoch, step_index, metrics_payload)

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

        zero_like = loss_focal.new_tensor(0.0)
        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
                'loss_aux': zero_like,
                'loss_dice': loss_dice if loss_dice is not None else zero_like,
                'loss_boundary': loss_boundary if loss_boundary is not None else zero_like,
            }

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

            # Tensorboard logger (skip for test split - no labels available)
            if not self.settings.test_split:
                tensorboard_logger(epoch=epoch,
                                   mode=mode,
                                   recorder=self.recorder,
                                   metrics_dict=metrics_dict,
                                   loss_dict=loss_dict,
                                   lr=lr_value,
                                   mapped_cls_name=self.mapped_cls_name)

                # Results at the end of the epoch
                log_str = '>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall {:0.4f}'.format(
                    mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
                self.recorder.logger.info(log_str)
            else:
                self.recorder.logger.info(f'>>> {mode} on test split completed - predictions saved')


        # Return metrics only if labels are available
        if not self.settings.test_split:
            result_metrics = {
                'Acc': mean_acc.item(),
                'IOU': mean_iou.item(),
                'Recall': mean_recall.item()
            }
            return result_metrics
        else:
            # Test split has no labels, return None or dummy metrics
            return None
