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
from typing import Optional, Callable, Dict

from option import Option
import dataset
import utils
import utils.tools as tools

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
            return_uproj=(not self.settings.use_kpconv),
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
        from utils.optim.boundary_loss import BoundaryLoss
        from utils.optim.focal_softmax import FocalSoftmaxLoss
        # Initialize focal loss (assuming number of classes is known from settings)
        num_classes = self.settings.num_classes if hasattr(self.settings, 'num_classes') else 20
        criterion['focal_loss'] = FocalSoftmaxLoss(num_classes)
        criterion['boundary'] = BoundaryLoss(self.settings.num_classes)
        # Set device
        for _, v in criterion.items():
            v.cuda()

        # Set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    def compute_losses(self, output, output_softmax, label, mask):
        loss_lovasz = self.criterion['lovasz'](output_softmax, label)
        loss_focal = self.criterion['focal_loss'](output_softmax, label, mask=mask)
        loss_boundary = self.criterion['boundary'](output_softmax, label)
        total_loss = loss_focal + loss_lovasz + loss_boundary
        return total_loss, loss_lovasz, loss_focal


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
        total_iter = len(dataloader)
        t_start = time.time()

        for i, batch_data in enumerate(dataloader):
            t_process_start = time.time()
            if len(batch_data) == 8:
                 input_feature, input_label, input_mask, proj_range, uproj_x, uproj_y, uproj_depth, labels_3d = batch_data
            else:
                 input_feature, input_label, input_mask = batch_data
                 labels_3d = None

            # Feature: range, x, y, z, intensity
            input_feature = input_feature.cuda() # shape: B x 5 x H x W

            input_label = input_label.cuda().long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.cuda() * input_label.ge(1).float()

            # Forward propagation
            if mode == 'Train':
                with torch.amp.autocast('cuda', enabled=self.fp16_scaler is not None):
                    output_tuple = self.model(input_feature)
                    if isinstance(output_tuple, tuple):
                        output, auxs = output_tuple
                    else:
                        output, auxs = output_tuple, None
                    
                    output_softmax = F.softmax(output, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal = self.compute_losses(
                        output, output_softmax, input_label, input_mask)
                    
                    # Auxiliary Loss
                    if auxs is not None:
                        for aux_logit in auxs:
                            aux_softmax = F.softmax(aux_logit, dim=1)
                            loss_aux, _, _ = self.compute_losses(aux_logit, aux_softmax, input_label, input_mask)
                            total_loss += 0.4 * loss_aux

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

            # Forward propagation
            if mode == 'Train':
                # ... (Train loop remains same, assuming loader returns 3 items for train)
                # Wait, train loader is NOT changed to return_uproj=True.
                pass
            
            # Validation
            else:
                # Unpack data
                # Loader returns 8 items if return_uproj=True
                if len(batch_data) == 8:
                    input_feature, input_label, input_mask, proj_range, uproj_x, uproj_y, uproj_depth, labels_3d = batch_data
                    input_feature = input_feature.cuda()
                    input_label = input_label.cuda().long()
                    input_mask = input_mask.cuda()
                    # uproj data needed for post-proc
                else:
                    # Fallback for legacy or if return_uproj failed
                    input_feature, input_label, input_mask = batch_data
                    input_feature = input_feature.cuda()
                    input_label = input_label.cuda().long()
                    input_mask = input_mask.cuda()
                    labels_3d = None

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

                # Loss calculation (2D)
                total_loss, loss_lovasz, loss_focal = self.compute_losses(
                    output, output_softmax, input_label, input_mask)

                # Measure IoU and record loss
                loss = total_loss.mean()
                
                with torch.no_grad():
                    argmax = output.argmax(dim=1)
                    
                    if labels_3d is not None and self.post_proc is not None:
                        # Post-processing (KNN)
                        # KNN expects unbatched input, but batch_size_val is usually 1.
                        # We iterate over batch just in case.
                        for b in range(output.shape[0]):
                            # KNN forward(proj_range, unproj_range, proj_argmax, px, py)
                            # proj_range: (H, W)
                            # unproj_range: (N,) -> uproj_depth
                            # proj_argmax: (H, W)
                            # px: uproj_x
                            # py: uproj_y
                            
                            pred_3d = self.post_proc(
                                proj_range[b].cuda(),
                                uproj_depth[b].cuda(),
                                argmax[b],
                                uproj_x[b].cuda(),
                                uproj_y[b].cuda()
                            )
                            
                            # Update metrics with 3D data
                            self.metrics.addBatch(pred_3d, labels_3d[b].cuda())
                    else:
                        # Fallback to 2D metrics if no 3D info
                        self.metrics.addBatch(argmax, input_label)

            loss_meter.update(loss.item(), input_feature.size(0))

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

        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
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

            # Tensorboard logger (skip for test split - no labels available)
            if not self.settings.test_split:
                tensorboard_logger(epoch=epoch,
                                   mode=mode,
                                   recorder=self.recorder,
                                   metrics_dict=metrics_dict,
                                   loss_dict=loss_dict,
                                   lr=lr,
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
            # assert self.settings.in_channels == 5

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
                    output3d_tuple = self.model(input_feature, px, py, pxyz, knns, num_points)
                    if isinstance(output3d_tuple, tuple):
                        output3d, _ = output3d_tuple # Ignore auxs for KPConv mode for now
                    else:
                        output3d = output3d_tuple

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal = self.compute_losses(
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
                    total_loss, loss_lovasz, loss_focal = self.compute_losses(
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

        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_lovasz': loss_lovasz,
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
