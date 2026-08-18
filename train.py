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
from utils.robust_eval import apply_robust_eval
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

        # Range image-level augmentation (applied on batch before forward pass)
        self.range_aug = None
        if getattr(self.settings, 'range_aug', False):
            from dataset.preprocess.rangeaug import RangeAugmentation
            self.range_aug = RangeAugmentation()

        if getattr(self.settings, 'robust_eval_enabled', False) and self.recorder is not None:
            self.recorder.logger.info(
                'Validation robustness enabled: type=%s severity=%.4f seed=%d',
                self.settings.robust_eval_type,
                self.settings.robust_eval_severity,
                self.settings.robust_eval_seed)

        # Define scheduler
        accum_steps = max(1, getattr(self.settings, 'grad_accum_steps', 1))
        steps_per_epoch = max(1, (len(self.train_loader) + accum_steps - 1) // accum_steps)
        self.scheduler = utils.optim.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            min_lr=getattr(self.settings, 'min_lr', 0.0),
            warmup_steps=self.settings.warmup_epochs * steps_per_epoch,
            momentum=0.9,
            max_steps=steps_per_epoch * (self.settings.n_epochs - self.settings.warmup_epochs))

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

            # Label-efficient (BALViT protocol): apply a percentile split at
            # the parser level via percentiles_split.json, plus optional
            # per-loader subsampling via dataset_skip_step / repeat_factor.
            skip_ratio = 1
            if self.settings.label_efficient_enable:
                skip_ratio = self.settings.dataset_skip_step_org

            trainset = dataset.semantic_kitti.SemanticKitti(
                root=self.settings.data_root,
                sequences=train_sequences,
                config_path=data_config_path,
                split='train',
                skip_ratio=skip_ratio)

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

        # SemanticPOSS dataset
        elif self.settings.dataset == 'SemanticPOSS':
            data_config_path = 'dataset/semantic_poss/semantic-poss.yaml'
            data_config = yaml.safe_load(open(data_config_path, 'r'))

            if self.settings.use_mini_version:
                train_sequences = [0]
            elif self.settings.use_trainval:
                train_sequences = data_config['split']['train'] + data_config['split']['valid']
            else:
                train_sequences = data_config['split']['train']

            trainset = dataset.semantic_poss.SemanticPOSS(
                root=self.settings.data_root,
                sequences=train_sequences,
                config_path=data_config_path,
                split='train',
                skip_ratio=1)

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

            valset = dataset.semantic_poss.SemanticPOSS(
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
            use_kpconv=self.settings.use_kpconv,
            dataset_skip_step=self.settings.dataset_skip_step,
            repeat_factor=self.settings.repeat_factor)

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

        if self.settings.focal_loss_type == 'class_weighted_focal':
            if self.settings.class_weights is not None:
                alpha = np.asarray(self.settings.class_weights, dtype=np.float32)
            elif self.settings.dataset in ('SemanticKitti', 'SemanticPOSS'):
                alpha = self.cls_weight.astype(np.float32)
            elif self.settings.dataset == 'nuScenes':
                alpha = np.ones((self.settings.n_classes), dtype=np.float32)
            else:
                alpha = np.ones((self.settings.n_classes), dtype=np.float32)
        else:
            if self.settings.dataset in ('SemanticKitti', 'SemanticPOSS'):
                alpha = np.log(1 + self.cls_weight)
                alpha = alpha / max(alpha.max(), 1e-6)
            elif self.settings.dataset == 'nuScenes':
                alpha = np.ones((self.settings.n_classes))
            else:
                alpha = np.ones((self.settings.n_classes))

        alpha[self.settings.focal_ignore_index] = 0
        if self.recorder is not None:
            self.recorder.logger.info('focal_loss_type: {}'.format(self.settings.focal_loss_type))
            self.recorder.logger.info('focal_loss_weight: {}'.format(self.settings.focal_loss_weight))
            self.recorder.logger.info('lovasz_loss_weight: {}'.format(self.settings.lovasz_loss_weight))
            self.recorder.logger.info('boundary_loss_weight: {}'.format(self.boundary_loss_weight))
            self.recorder.logger.info('aux_loss_weight: {}'.format(self.settings.aux_loss_weight))
            self.recorder.logger.info('focal_gamma: {}'.format(self.settings.focal_gamma))
            self.recorder.logger.info('focal_ignore_index: {}'.format(self.settings.focal_ignore_index))
            self.recorder.logger.info('focal_loss alpha: {}'.format(alpha))

        criterion['focal_loss'] = utils.optim.FocalSoftmaxLoss(
            self.settings.n_classes,
            gamma=self.settings.focal_gamma,
            alpha=alpha,
            softmax=False,
            ignore_index=self.settings.focal_ignore_index)
        criterion['boundary_loss'] = utils.optim.BoundaryLoss(
            ignore_index=self.settings.focal_ignore_index)

        # Set device
        for _, v in criterion.items():
            v.cuda()
        return criterion

    def _split_main_and_aux_output(self, output):
        if isinstance(output, (tuple, list)):
            main = output[0]
            aux_outputs = []
            if len(output) > 1:
                aux_item = output[1]
                if isinstance(aux_item, (tuple, list)):
                    aux_outputs = [a for a in aux_item if torch.is_tensor(a)]
                elif torch.is_tensor(aux_item):
                    aux_outputs = [aux_item]
            return main, aux_outputs
        return output, []

    def _build_tta_metas(self, input_feature):
        tta_mode = str(getattr(self.settings, 'tta', 'none')).lower()
        if not (getattr(self.settings, 'val_only', False)):
            return [dict(flip=False, shift=0)]
        width = int(input_feature.shape[-1])

        def unique_shifts(values):
            seen = set()
            ordered = []
            for val in values:
                shift = int(val) % max(width, 1)
                if shift > width // 2:
                    shift -= width
                if shift not in seen:
                    seen.add(shift)
                    ordered.append(shift)
            return ordered

        shifts = [0]
        flips = [False]
        if tta_mode == 'flip':
            flips = [False, True]
        elif tta_mode == 'shift':
            shifts = unique_shifts([0, width // 4, width // 2, -width // 4])
        elif tta_mode == 'strong':
            shifts = unique_shifts([0, width // 4, width // 2, -width // 4])
            flips = [False, True]

        metas = []
        for shift in shifts:
            for flip in flips:
                metas.append(dict(flip=flip, shift=shift))
        return metas

    def _get_validation_sample_id(self, sample_index):
        try:
            sample_index = int(sample_index)
        except (TypeError, ValueError):
            return str(sample_index)

        range_dataset = getattr(self.val_range_loader, 'dataset', None)
        base_dataset = getattr(range_dataset, 'dataset', None)

        if range_dataset is not None and hasattr(range_dataset, '_get_sample_id'):
            return os.path.normpath(str(range_dataset._get_sample_id(sample_index)))
        if base_dataset is not None and hasattr(base_dataset, 'parsePathInfoByIndex'):
            seq_id, frame_id = base_dataset.parsePathInfoByIndex(sample_index)
            return os.path.join(str(seq_id), str(frame_id))
        if base_dataset is not None and hasattr(base_dataset, 'lidar_filenames'):
            return os.path.normpath(str(base_dataset.lidar_filenames[sample_index]))
        if base_dataset is not None and hasattr(base_dataset, 'pointcloud_files'):
            return os.path.normpath(str(base_dataset.pointcloud_files[sample_index]))
        return str(sample_index)

    def compute_losses(self, output, output_softmax, label, mask, aux_outputs=None, aux_loss_weight=0.0):
        loss_lovasz = self.criterion['lovasz'](output_softmax, label)
        if torch.is_tensor(loss_lovasz) and loss_lovasz.ndim > 0:
            loss_lovasz = loss_lovasz.mean()
        loss_focal = self.criterion['focal_loss'](output_softmax, label, mask=mask)
        if torch.is_tensor(loss_focal) and loss_focal.ndim > 0:
            loss_focal = loss_focal.mean()
        total_loss = (
            self.settings.focal_loss_weight * loss_focal +
            self.settings.lovasz_loss_weight * loss_lovasz
        )

        loss_boundary = torch.zeros([], device=output.device, dtype=output.dtype)
        if self.boundary_loss_weight > 0:
            loss_boundary = self.criterion['boundary_loss'](output_softmax, label, mask)
            if torch.is_tensor(loss_boundary) and loss_boundary.ndim > 0:
                loss_boundary = loss_boundary.mean()
            total_loss = total_loss + self.boundary_loss_weight * loss_boundary

        loss_aux = torch.zeros([], device=output.device, dtype=output.dtype)
        if aux_outputs and aux_loss_weight > 0:
            aux_total = torch.zeros([], device=output.device, dtype=output.dtype)
            for aux_out in aux_outputs:
                aux_softmax = F.softmax(aux_out, dim=1)
                aux_lovasz = self.criterion['lovasz'](aux_softmax, label)
                if torch.is_tensor(aux_lovasz) and aux_lovasz.ndim > 0:
                    aux_lovasz = aux_lovasz.mean()
                aux_focal = self.criterion['focal_loss'](aux_softmax, label, mask=mask)
                if torch.is_tensor(aux_focal) and aux_focal.ndim > 0:
                    aux_focal = aux_focal.mean()
                aux_loss = (
                    self.settings.lovasz_loss_weight * aux_lovasz +
                    self.settings.focal_loss_weight * aux_focal
                )
                if self.boundary_loss_weight > 0:
                    aux_boundary = self.criterion['boundary_loss'](aux_softmax, label, mask)
                    if torch.is_tensor(aux_boundary) and aux_boundary.ndim > 0:
                        aux_boundary = aux_boundary.mean()
                    aux_loss = aux_loss + self.boundary_loss_weight * aux_boundary
                aux_total = aux_total + aux_loss
            loss_aux = aux_total
            if torch.is_tensor(loss_aux) and loss_aux.ndim > 0:
                loss_aux = loss_aux.mean()
            total_loss = total_loss + aux_loss_weight * loss_aux

        if torch.is_tensor(total_loss) and total_loss.ndim > 0:
            total_loss = total_loss.mean()

        return total_loss, loss_lovasz, loss_focal, loss_boundary, loss_aux


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

        track_mem = torch.cuda.is_available()
        if track_mem:
            torch.cuda.reset_peak_memory_stats()
            mem_device = torch.device('cuda')
            run_peak_alloc_bytes = 0
            run_peak_res_bytes = 0
        else:
            mem_device = None
            run_peak_alloc_bytes = None
            run_peak_res_bytes = None
        infer_mem_count = 0
        infer_alloc_peak_sum = 0
        infer_res_peak_sum = 0
        infer_alloc_delta_sum = 0
        infer_res_delta_sum = 0
        infer_alloc_peak_max = 0
        infer_res_peak_max = 0
        infer_alloc_delta_max = 0
        infer_res_delta_max = 0

        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module

        # Init metrics
        loss_meter = tools.AverageMeter()
        loss_focal_meter = tools.AverageMeter()
        loss_lovasz_meter = tools.AverageMeter()
        loss_boundary_w_meter = tools.AverageMeter()
        loss_aux_w_meter = tools.AverageMeter()
        self.metrics.reset()
        if mode == 'Validation' and self.use_knn:
            self.metrics_3d.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        if mode == 'Train':
            self.optimizer.zero_grad()

        log_frequency = max(1, self.settings.log_frequency)

        for i, batch in enumerate(dataloader):
            t_process_start = time.time()
            current_lr = None
            sample_index = i
            if mode == 'Validation' and self.use_knn:
                if len(batch) == 9:
                    (input_feature, input_label, input_mask, proj_depth,
                     uproj_x_idx, uproj_y_idx, uproj_depth, sem_label, sample_index) = batch
                    if torch.is_tensor(sample_index):
                        sample_index = int(sample_index.item())
                else:
                    (input_feature, input_label, input_mask, proj_depth,
                     uproj_x_idx, uproj_y_idx, uproj_depth, sem_label) = batch
            else:
                input_feature, input_label, input_mask = batch

            # Feature: range, x, y, z, intensity
            input_feature = input_feature.cuda() # shape: B x 5 x H x W

            input_label = input_label.cuda().long()
            input_label = input_label * input_label.ge(1).long()
            input_mask = input_mask.cuda() * input_label.ge(1).float()

            # Range image-level augmentation
            if mode == 'Train' and self.range_aug is not None:
                input_feature, input_label = self.range_aug(input_feature, input_label, input_mask)
            if mode == 'Validation' and getattr(self.settings, 'robust_eval_enabled', False):
                input_feature = apply_robust_eval(
                    input_feature=input_feature,
                    input_mask=input_mask,
                    corruption_type=self.settings.robust_eval_type,
                    severity=self.settings.robust_eval_severity,
                    seed=int(self.settings.robust_eval_seed) + int(sample_index))

            # Forward propagation
            if mode == 'Train':
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output = self.model(input_feature)
                    output, aux_outputs = self._split_main_and_aux_output(output)
                    output_softmax = F.softmax(output, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary, loss_aux = self.compute_losses(
                        output, output_softmax, input_label, input_mask,
                        aux_outputs=aux_outputs,
                        aux_loss_weight=self.settings.aux_loss_weight)

                # Backward with gradient accumulation
                accum_steps = max(1, getattr(self.settings, 'grad_accum_steps', 1))
                loss_scaled = total_loss / accum_steps
                if self.fp16_scaler is None:
                    loss_scaled.backward()
                else:
                    self.fp16_scaler.scale(loss_scaled).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == total_iter:
                    if self.fp16_scaler is None:
                        self.optimizer.step()
                    else:
                        self.fp16_scaler.step(self.optimizer)
                        self.fp16_scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            with torch.no_grad():
                if mode == 'Validation':
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    ims_metas = self._build_tta_metas(input_feature)
                    ims = [input_feature for _ in ims_metas]
                    if track_mem:
                        run_peak_alloc_bytes = max(
                            run_peak_alloc_bytes,
                            torch.cuda.max_memory_allocated(mem_device))
                        run_peak_res_bytes = max(
                            run_peak_res_bytes,
                            torch.cuda.max_memory_reserved(mem_device))
                        torch.cuda.reset_peak_memory_stats()
                        infer_start_alloc = torch.cuda.memory_allocated(mem_device)
                        infer_start_res = torch.cuda.memory_reserved(mem_device)
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        lidar_pred = inference(
                            model_without_ddp.rangevit,
                            ims,
                            ims_metas,
                            ori_shape=input_feature.shape[2:4],
                            window_size=self.settings.window_size,
                            window_stride=self.settings.window_stride,
                            batch_size=input_feature.shape[0],
                            use_kpconv=False,
                            use_sliding_window=self.settings.use_sliding_window)
                    if track_mem:
                        infer_peak_alloc = torch.cuda.max_memory_allocated(mem_device)
                        infer_peak_res = torch.cuda.max_memory_reserved(mem_device)
                        infer_delta_alloc = max(0, infer_peak_alloc - infer_start_alloc)
                        infer_delta_res = max(0, infer_peak_res - infer_start_res)
                        run_peak_alloc_bytes = max(run_peak_alloc_bytes, infer_peak_alloc)
                        run_peak_res_bytes = max(run_peak_res_bytes, infer_peak_res)
                        infer_mem_count += 1
                        infer_alloc_peak_sum += infer_peak_alloc
                        infer_res_peak_sum += infer_peak_res
                        infer_alloc_delta_sum += infer_delta_alloc
                        infer_res_delta_sum += infer_delta_res
                        infer_alloc_peak_max = max(infer_alloc_peak_max, infer_peak_alloc)
                        infer_res_peak_max = max(infer_res_peak_max, infer_peak_res)
                        infer_alloc_delta_max = max(infer_alloc_delta_max, infer_delta_alloc)
                        infer_res_delta_max = max(infer_res_delta_max, infer_delta_res)

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
                    if self.settings.has_label:
                        total_loss, loss_lovasz, loss_focal, loss_boundary, loss_aux = self.compute_losses(
                            output, output_softmax, input_label, input_mask)
                    else:
                        zero = torch.tensor(0.0, device=output.device)
                        total_loss = loss_lovasz = loss_focal = loss_boundary = loss_aux = zero

            current_lr = self.optimizer.param_groups[0]['lr']

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax = output.argmax(dim=1)
                if self.settings.has_label:
                    self.metrics.addBatch(argmax, input_label) # 2D predictions (pixel metrics)
                    if mode == 'Validation' and self.use_knn:
                        self.metrics_3d.addBatch(unproj_argmax, sem_label) # 3D predictions (point metrics)

            loss_meter.update(loss.item(), input_feature.size(0))
            with torch.no_grad():
                loss_focal_val = float((self.settings.focal_loss_weight * loss_focal).detach())
                loss_lovasz_val = float((self.settings.lovasz_loss_weight * loss_lovasz).detach())
                loss_boundary_w_val = float((self.boundary_loss_weight * loss_boundary).detach())
                loss_aux_w_val = float((self.settings.aux_loss_weight * loss_aux).detach())
            loss_focal_meter.update(loss_focal_val, input_feature.size(0))
            loss_lovasz_meter.update(loss_lovasz_val, input_feature.size(0))
            loss_boundary_w_meter.update(loss_boundary_w_val, input_feature.size(0))
            loss_aux_w_meter.update(loss_aux_w_val, input_feature.size(0))

            with torch.no_grad():
                if self.settings.has_label:
                    mean_iou_tensor, _, mean_acc_tensor, _ = self.metrics.getIoUnAcc()
                    mean_recall_tensor, _ = self.metrics.getRecall()
                    mean_iou_point_tensor = mean_acc_point_tensor = mean_recall_point_tensor = torch.tensor(0.0)
                    if mode == 'Validation' and self.use_knn:
                        mean_iou_point_tensor, _, mean_acc_point_tensor, _ = self.metrics_3d.getIoUnAcc()
                        mean_recall_point_tensor, _ = self.metrics_3d.getRecall()
                else:
                    mean_iou_tensor = mean_acc_tensor = mean_recall_tensor = torch.tensor(0.0)
                    mean_iou_point_tensor = mean_acc_point_tensor = mean_recall_point_tensor = torch.tensor(0.0)
            mean_iou_running = float(mean_iou_tensor)
            mean_acc_running = float(mean_acc_tensor)
            mean_recall_running = float(mean_recall_tensor)
            mean_iou_point_running = float(mean_iou_point_tensor)
            mean_acc_point_running = float(mean_acc_point_tensor)
            mean_recall_point_running = float(mean_recall_point_tensor)

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
            should_log = (i % log_frequency == 0) or (i == total_iter - 1)

            # Logging
            if should_log:
                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} [foc {:.3f} lov {:.3f} bndW {:.3f} auxW {:.3f}] Acc {:0.4f} IOU {:0.4F} '.format(
                        current_lr, loss.item(), loss_focal_val, loss_lovasz_val, loss_boundary_w_val, loss_aux_w_val,
                        mean_acc_running, mean_iou_running)
                    if mode == 'Validation' and self.use_knn:
                        log_str += 'Acc_point {:0.4f} IOU_point {:0.4F} '.format(
                            mean_acc_point_running, mean_iou_point_running)
                    log_str += 'RT {}'.format(remain_time)
                    self.recorder.logger.info(log_str)

        with torch.no_grad():
            if self.settings.has_label:
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
                mean_acc_point = metrics_dict_3d['mean_acc'].item() if metrics_dict_3d is not None else 0.0
                mean_recall_point = metrics_dict_3d['mean_recall'].item() if metrics_dict_3d is not None else 0.0
                mean_iou_point = metrics_dict_3d['mean_iou'].item() if metrics_dict_3d is not None else 0.0
            else:
                zero_t = torch.tensor(0.0)
                mean_acc = mean_recall = mean_iou = zero_t
                metrics_dict = {
                    'mean_acc': zero_t,
                    'class_acc': None,
                    'mean_recall': zero_t,
                    'class_recall': None,
                    'mean_iou': zero_t,
                    'class_iou': None,
                    'conf_matrix': None,
                }
                metrics_dict_3d = None

        loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal_meter.avg,
                'loss_lovasz': loss_lovasz_meter.avg,
                'loss_boundary_weighted': loss_boundary_w_meter.avg,
                'loss_aux_weighted': loss_aux_w_meter.avg,
                'loss_component_sum_weighted': (
                    loss_focal_meter.avg + loss_lovasz_meter.avg + loss_boundary_w_meter.avg + loss_aux_w_meter.avg
                ),
            }

        epoch_lr = self.optimizer.param_groups[0]['lr']
        max_alloc_mb = max_res_mb = None
        if track_mem:
            run_peak_alloc_bytes = max(
                run_peak_alloc_bytes,
                torch.cuda.max_memory_allocated(mem_device))
            run_peak_res_bytes = max(
                run_peak_res_bytes,
                torch.cuda.max_memory_reserved(mem_device))
            max_alloc_mb = run_peak_alloc_bytes / (1024 ** 2)
            max_res_mb = run_peak_res_bytes / (1024 ** 2)
        infer_mem_summary = None
        if infer_mem_count > 0:
            mb = 1024 ** 2
            infer_mem_summary = {
                'count': infer_mem_count,
                'avg_alloc_peak': (infer_alloc_peak_sum / infer_mem_count) / mb,
                'avg_res_peak': (infer_res_peak_sum / infer_mem_count) / mb,
                'avg_alloc_delta': (infer_alloc_delta_sum / infer_mem_count) / mb,
                'avg_res_delta': (infer_res_delta_sum / infer_mem_count) / mb,
                'max_alloc_peak': infer_alloc_peak_max / mb,
                'max_res_peak': infer_res_peak_max / mb,
                'max_alloc_delta': infer_alloc_delta_max / mb,
                'max_res_delta': infer_res_delta_max / mb,
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

            # Print validation pixel-wise evaluation results (only when labels available)
            if self.settings.has_label and mode == 'Validation' and (print_results or epoch == self.settings.n_epochs-1):
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

            # Tensorboard logger (guarded when labels exist)
            if self.settings.has_label:
                tensorboard_logger(epoch=epoch,
                                   mode=mode,
                                   recorder=self.recorder,
                                   metrics_dict=metrics_dict,
                                   loss_dict=loss_dict,
                                   lr=epoch_lr,
                                   mapped_cls_name=self.mapped_cls_name)

            # Results at the end of the epoch
            log_str = '>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4F} Recall_pixel {:0.4f}'.format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            if mode == 'Validation' and self.use_knn and metrics_dict_3d is not None:
                log_str += ' | Acc_point {:0.4f} IOU_point {:0.4F} Recall_point {:0.4f}'.format(
                    mean_acc_point, mean_iou_point, mean_recall_point)
            if max_alloc_mb is not None and max_res_mb is not None:
                log_str += ' | Mem_alloc {:.1f}MB Mem_res {:.1f}MB'.format(max_alloc_mb, max_res_mb)
            self.recorder.logger.info(log_str)
            if mode == 'Validation' and infer_mem_summary is not None:
                self.recorder.logger.info(
                    '>>> Validation InferMemAvg N[{count}] '
                    'Mem_alloc_peak_avg {avg_alloc_peak:.1f}MB Mem_res_peak_avg {avg_res_peak:.1f}MB '
                    'Mem_alloc_delta_avg {avg_alloc_delta:.1f}MB Mem_res_delta_avg {avg_res_delta:.1f}MB '
                    'Mem_alloc_peak_max {max_alloc_peak:.1f}MB Mem_res_peak_max {max_res_peak:.1f}MB '
                    'Mem_alloc_delta_max {max_alloc_delta:.1f}MB Mem_res_delta_max {max_res_delta:.1f}MB'.format(
                        **infer_mem_summary))

        # Prefer point metrics for best-model selection when available (KNN validation),
        # otherwise fall back to pixel metrics.
        primary_acc = mean_acc_point if (self.settings.has_label and metrics_dict_3d is not None) else mean_acc
        primary_iou = mean_iou_point if (self.settings.has_label and metrics_dict_3d is not None) else mean_iou
        primary_recall = mean_recall_point if (self.settings.has_label and metrics_dict_3d is not None) else mean_recall

        if self.mlflow_manager is not None:
            mlflow_metrics = {
                f'{mode.lower()}_loss': loss_meter.avg,
                f'{mode.lower()}_loss_focal': loss_focal_meter.avg,
                f'{mode.lower()}_loss_lovasz': loss_lovasz_meter.avg,
                f'{mode.lower()}_loss_boundary_weighted': loss_boundary_w_meter.avg,
                f'{mode.lower()}_loss_aux_weighted': loss_aux_w_meter.avg,
                f'{mode.lower()}_acc': primary_acc if isinstance(primary_acc, float) else primary_acc.item(),
                f'{mode.lower()}_recall': primary_recall if isinstance(primary_recall, float) else primary_recall.item(),
                f'{mode.lower()}_iou_pixel': mean_iou.item(),
            }
            if max_alloc_mb is not None and max_res_mb is not None:
                mlflow_metrics[f'{mode.lower()}_max_mem_alloc_mb'] = max_alloc_mb
                mlflow_metrics[f'{mode.lower()}_max_mem_reserved_mb'] = max_res_mb
            if mode == 'Validation' and self.use_knn and metrics_dict_3d is not None:
                mlflow_metrics[f'{mode.lower()}_iou_point'] = mean_iou_point
            if (mode == 'Train') and (epoch_lr is not None):
                mlflow_metrics[f'{mode.lower()}_lr'] = epoch_lr
            ignored_classes = set(getattr(self, 'ignore_class', []))
            class_iou_for_log = metrics_dict.get('class_iou', None)
            if class_iou_for_log is not None:
                for cls_id, cls_iou in enumerate(class_iou_for_log):
                    if cls_id in ignored_classes:
                        continue
                    cls_name = None
                    if isinstance(self.mapped_cls_name, dict):
                        cls_name = self.mapped_cls_name.get(cls_id, None)
                    elif isinstance(self.mapped_cls_name, (list, tuple)) and cls_id < len(self.mapped_cls_name):
                        cls_name = self.mapped_cls_name[cls_id]
                    if cls_name is None:
                        cls_name = f'class_{cls_id}'
                    safe_name = ''.join(ch if ch.isalnum() else '_' for ch in str(cls_name).lower()).strip('_')
                    if not safe_name:
                        safe_name = f'class_{cls_id}'
                    # Name by class first so train/validation curves of the same class stay adjacent.
                    mlflow_metrics[f'zz_{safe_name}_iou_{mode.lower()}'] = float(cls_iou.item())
            # Skip logging label-dependent metrics when labels are absent.
            if self.settings.has_label:
                self.mlflow_manager.log_metrics(mlflow_metrics, step=epoch + 1)

        result_metrics = {
            'macc': primary_acc if isinstance(primary_acc, float) else primary_acc.item(),
            'miou': primary_iou if isinstance(primary_iou, float) else primary_iou.item(),
            'mrecall': primary_recall if isinstance(primary_recall, float) else primary_recall.item(),
            'miou_pixel': mean_iou.item(),
            # Legacy aliases (compatibility)
            'Acc': primary_acc if isinstance(primary_acc, float) else primary_acc.item(),
            'IOU': primary_iou if isinstance(primary_iou, float) else primary_iou.item(),
            'Recall': primary_recall if isinstance(primary_recall, float) else primary_recall.item(),
            'IOU_pixel': mean_iou.item(),
        }
        if metrics_dict_3d is not None:
            result_metrics.update({
                'miou_point': mean_iou_point,
                # Legacy aliases (compatibility)
                'IOU_point': mean_iou_point,
            })

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

        track_mem = torch.cuda.is_available()
        if track_mem:
            torch.cuda.reset_peak_memory_stats()
            mem_device = torch.device('cuda')
        else:
            mem_device = None

        track_remain_time_1epoch = tools.RemainTime(1)

        model_without_ddp = self.model
        if hasattr(self.model, 'module'):
            model_without_ddp = self.model.module

        # Init metrics
        loss_meter = tools.AverageMeter()
        loss_focal_meter = tools.AverageMeter()
        loss_lovasz_meter = tools.AverageMeter()
        loss_boundary_w_meter = tools.AverageMeter()
        loss_aux_w_meter = tools.AverageMeter()
        self.metrics.reset()

        total_iter = len(dataloader)
        t_start = time.time()

        if mode == 'Train':
            self.optimizer.zero_grad()

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

            # Range image-level augmentation (on 2D range image before 3D processing)
            if mode == 'Train' and self.range_aug is not None:
                input_feature, _ = self.range_aug(
                    input_feature,
                    labels3d.squeeze(1).squeeze(1),
                    mask_3d.squeeze(1).squeeze(1))

            # Forward propagation
            if mode == 'Train':
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output3d = self.model(input_feature, px, py, pxyz, knns, num_points)

                    output3d_softmax = F.softmax(output3d, dim=1)

                    # Loss calculation
                    total_loss, loss_lovasz, loss_focal, loss_boundary, loss_aux = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

                # Backward with gradient accumulation
                accum_steps = max(1, getattr(self.settings, 'grad_accum_steps', 1))
                loss_scaled = total_loss / accum_steps
                if self.fp16_scaler is None:
                    loss_scaled.backward()
                else:
                    self.fp16_scaler.scale(loss_scaled).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == total_iter:
                    if self.fp16_scaler is None:
                        self.optimizer.step()
                    else:
                        self.fp16_scaler.step(self.optimizer)
                        self.fp16_scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            with torch.no_grad():
                if mode == 'Validation':
                    assert input_feature.shape[0] == 1 # validation batch size has to be 1

                    # Validation
                    ims_metas = self._build_tta_metas(input_feature)
                    ims = [input_feature for _ in ims_metas]
                    with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                        output_features2d = inference(
                            model_without_ddp.rangevit,
                            ims,
                            ims_metas,
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
                    total_loss, loss_lovasz, loss_focal, loss_boundary, loss_aux = self.compute_losses(
                        output3d, output3d_softmax, labels3d, mask_3d)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Measure IoU and record loss
            loss = total_loss.mean()
            with torch.no_grad():
                argmax3d = output3d.argmax(dim=1)
                self.metrics.addBatch(argmax3d, labels3d) # 3D predictions

            loss_meter.update(loss.item(), input_feature.size(0))
            with torch.no_grad():
                loss_focal_val = float((self.settings.focal_loss_weight * loss_focal).detach())
                loss_lovasz_val = float((self.settings.lovasz_loss_weight * loss_lovasz).detach())
                loss_boundary_w_val = float((self.boundary_loss_weight * loss_boundary).detach())
                loss_aux_w_val = float((self.settings.aux_loss_weight * loss_aux).detach())
            loss_focal_meter.update(loss_focal_val, input_feature.size(0))
            loss_lovasz_meter.update(loss_lovasz_val, input_feature.size(0))
            loss_boundary_w_meter.update(loss_boundary_w_val, input_feature.size(0))
            loss_aux_w_meter.update(loss_aux_w_val, input_feature.size(0))

            with torch.no_grad():
                mean_iou_tensor, _, mean_acc_tensor, _ = self.metrics.getIoUnAcc()
                mean_recall_tensor, _ = self.metrics.getRecall()
            mean_iou_running = float(mean_iou_tensor)
            mean_acc_running = float(mean_acc_tensor)
            mean_recall_running = float(mean_recall_tensor)

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

                elif self.settings.dataset in ('SemanticKitti', 'SemanticPOSS'):
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
            should_log = (i % log_frequency == 0) or (i == total_iter - 1)

            # Logging
            if should_log:
                if self.recorder is not None:
                    log_str = '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] '.format(
                        mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
                    log_str += 'LR {} Loss {:0.4f} [foc {:.3f} lov {:.3f} bndW {:.3f} auxW {:.3f}] Acc_point {:0.4f} IOU_point {:0.4F} '.format(
                        current_lr, loss.item(), loss_focal_val, loss_lovasz_val, loss_boundary_w_val, loss_aux_w_val,
                        mean_acc_running, mean_iou_running)
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
                'loss_focal': loss_focal_meter.avg,
                'loss_lovasz': loss_lovasz_meter.avg,
                'loss_boundary_weighted': loss_boundary_w_meter.avg,
                'loss_aux_weighted': loss_aux_w_meter.avg,
                'loss_component_sum_weighted': (
                    loss_focal_meter.avg + loss_lovasz_meter.avg + loss_boundary_w_meter.avg + loss_aux_w_meter.avg
                ),
            }

        epoch_lr = self.optimizer.param_groups[0]['lr']
        max_alloc_mb = max_res_mb = None
        if track_mem:
            max_alloc_mb = torch.cuda.max_memory_allocated(mem_device) / (1024 ** 2)
            max_res_mb = torch.cuda.max_memory_reserved(mem_device) / (1024 ** 2)

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
            log_str = '>>> {} Loss {:0.4f} Acc_point {:0.4f} IOU_point {:0.4F} Recall_point {:0.4f}'.format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item())
            if max_alloc_mb is not None and max_res_mb is not None:
                log_str += ' | Mem_alloc {:.1f}MB Mem_res {:.1f}MB'.format(max_alloc_mb, max_res_mb)
            self.recorder.logger.info(log_str)

        if self.mlflow_manager is not None:
            mlflow_metrics = {
                f'{mode.lower()}_loss': loss_meter.avg,
                f'{mode.lower()}_loss_focal': loss_focal_meter.avg,
                f'{mode.lower()}_loss_lovasz': loss_lovasz_meter.avg,
                f'{mode.lower()}_loss_boundary_weighted': loss_boundary_w_meter.avg,
                f'{mode.lower()}_loss_aux_weighted': loss_aux_w_meter.avg,
                f'{mode.lower()}_acc': mean_acc.item(),
                f'{mode.lower()}_recall': mean_recall.item(),
                f'{mode.lower()}_iou_point': mean_iou.item(),
            }
            if max_alloc_mb is not None and max_res_mb is not None:
                mlflow_metrics[f'{mode.lower()}_max_mem_alloc_mb'] = max_alloc_mb
                mlflow_metrics[f'{mode.lower()}_max_mem_reserved_mb'] = max_res_mb
            if (mode == 'Train') and (epoch_lr is not None):
                mlflow_metrics[f'{mode.lower()}_lr'] = epoch_lr
            ignored_classes = set(getattr(self, 'ignore_class', []))
            class_iou_for_log = metrics_dict.get('class_iou', None)
            if class_iou_for_log is not None:
                for cls_id, cls_iou in enumerate(class_iou_for_log):
                    if cls_id in ignored_classes:
                        continue
                    cls_name = None
                    if isinstance(self.mapped_cls_name, dict):
                        cls_name = self.mapped_cls_name.get(cls_id, None)
                    elif isinstance(self.mapped_cls_name, (list, tuple)) and cls_id < len(self.mapped_cls_name):
                        cls_name = self.mapped_cls_name[cls_id]
                    if cls_name is None:
                        cls_name = f'class_{cls_id}'
                    safe_name = ''.join(ch if ch.isalnum() else '_' for ch in str(cls_name).lower()).strip('_')
                    if not safe_name:
                        safe_name = f'class_{cls_id}'
                    # Name by class first so train/validation curves of the same class stay adjacent.
                    mlflow_metrics[f'zz_{safe_name}_iou_{mode.lower()}'] = float(cls_iou.item())
            self.mlflow_manager.log_metrics(mlflow_metrics, step=epoch + 1)

        # KPConv path already works on point metrics; expose explicit aliases for clarity.
        result_metrics = {
            'macc': mean_acc.item(),
            'miou': mean_iou.item(),
            'mrecall': mean_recall.item(),
            'miou_point': mean_iou.item(),
            # Legacy aliases (compatibility)
            'Acc': mean_acc.item(),
            'IOU': mean_iou.item(),
            'Recall': mean_recall.item(),
            'IOU_point': mean_iou.item(),
        }

        return result_metrics
