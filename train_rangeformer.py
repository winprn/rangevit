# RangeFormer Training Script
# Extends RangeViT's training infrastructure with RangeFormer-specific features

import torch
import yaml
import os
import time
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from option import Option
import dataset
import utils
import utils.tools as tools
from utils.metrics.eval_results import eval_results
from utils.metrics.tensorboard_logger import tensorboard_logger
from utils.tools import Recorder
from utils.optim import DiceLoss, BoundaryLoss, Lovasz_softmax

# Import RangeFormer
from models.rangeformer import RangeFormer, create_rangeformer


class RangeFormerTrainer(object):
    """
    Trainer for RangeFormer model.

    Key differences from RangeViT trainer:
    - Handles auxiliary losses from multi-scale decoder
    - Supports RangeAug data augmentation
    - Uses 6-channel input (vs 5 for RangeViT)
    """

    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # Init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = tools.RemainTime(self.settings.n_epochs)
        self.prediction_path = None
        if self.settings.save_eval_results:
            base_path = self.recorder.save_path if self.recorder is not None else self.settings.save_path
            self.prediction_path = os.path.join(base_path, 'preds')
            if tools.is_main_process():
                os.makedirs(self.prediction_path, exist_ok=True)

        # Init data loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # Init criterion
        self.criterion = self._initCriterion()

        # Init optimizer
        self.optimizer = self._initOptimizer()

        # Distributed training
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

        # Define scheduler (OneCycle learning rate schedule per paper)
        onecycle_cfg = self.settings.config.get('onecycle', {})
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.settings.lr,
            epochs=self.settings.n_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=onecycle_cfg.get('pct_start', 0.1),
            div_factor=onecycle_cfg.get('div_factor', 25.0),
            final_div_factor=onecycle_cfg.get('final_div_factor', 1e4),
            anneal_strategy=onecycle_cfg.get('anneal_strategy', 'cos')
        )

        # For mixed precision training
        self.fp16_scaler = None
        if self.settings.use_fp16:
            print('Using mixed precision training (FP16)')
            raise NotImplementedError("[train_rangeformer] Mixed precision training is not implemented in current snippet.")
            self.fp16_scaler = torch.cuda.amp.GradScaler()

        # Auxiliary loss weight
        self.aux_loss_weight = self.settings.config.get('loss', {}).get('aux_loss_weight', 0.4)
        print(f'Auxiliary loss weight: {self.aux_loss_weight}')

        # STR configuration
        str_config = self.settings.config.get('str', {})
        self.use_str = bool(str_config.get('enabled', False))
        self.str_num_views = int(str_config.get('num_views', 1))
        self.str_align_inference = bool(str_config.get('align_inference', True))
        self.str_inference_views = int(str_config.get('inference_views', self.str_num_views))
        self.str_view_width = self.settings.image_size[1]
        self.str_full_width = self.settings.original_image_size[1]
        if self.use_str:
            if self.str_full_width % self.str_inference_views != 0:
                raise ValueError(
                    f'STR inference views ({self.str_inference_views}) must divide panorama width '
                    f'({self.str_full_width})')
            expected_width = self.str_full_width // self.str_num_views
            if self.str_view_width != expected_width:
                raise ValueError(
                    f'Config mismatch: image_size width ({self.str_view_width}) should equal '
                    f'original_image_size width / num_views ({expected_width})')

    def _initOptimizer(self):
        params = self.model.parameters()
        adamw_optimizer = torch.optim.AdamW(
            params=params,
            lr=self.settings.lr,
            weight_decay=self.settings.config.get('weight_decay', 0.01))
        return adamw_optimizer

    def _forward_with_str_views(self, range_images: torch.Tensor):
        """
        Run STR-aligned inference by slicing the panorama into view segments that
        match the training resolution and stitching logits back together.
        """
        if not self.use_str or not self.str_align_inference:
            logits, _ = self.model(range_images)
            return logits

        B, C, H, W_full = range_images.shape
        # If already at training width, run a single forward
        if W_full == self.str_view_width:
            logits, _ = self.model(range_images)
            return logits

        if W_full % self.str_view_width != 0:
            raise ValueError(
                f'STR inference requires panorama width ({W_full}) to be divisible '
                f'by view width ({self.str_view_width}).'
            )

        num_views = self.str_inference_views
        logits_views = []
        for view_idx in range(num_views):
            start = view_idx * self.str_view_width
            end = start + self.str_view_width
            view_tensor = range_images[:, :, :, start:end]
            logits_view, _ = self.model(view_tensor)
            logits_views.append(logits_view)

        logits_full = torch.cat(logits_views, dim=3)
        return logits_full

    def _initDataloader(self):
        """Initialize data loaders with RangeFormer loader."""
        # Import RangeFormer loader
        from dataset.rangeformer_loader import RangeFormerLoader

        # SemanticKITTI dataset
        if self.settings.dataset == 'SemanticKITTI':
            print('----Using SemanticKITTI dataset----')
            from dataset.semantic_kitti.parser import SemanticKitti

            data_config_path = self.settings.config.get(
                'data_config_path', 'dataset/semantic_kitti/semantic-kitti.yaml')
            data_config = yaml.safe_load(open(data_config_path, 'r'))

            if self.settings.use_mini_version:
                train_sequences = ['00']
            elif self.settings.use_trainval:
                train_sequences = data_config['split']['train'] + data_config['split']['valid']
            else:
                train_sequences = data_config['split']['train']

            train_sequences = [f'{int(seq):02d}' for seq in train_sequences]

            if self.settings.test_split:
                val_sequences = data_config['split']['test']
                val_has_label = False
            else:
                val_sequences = data_config['split']['valid']
                val_has_label = True

            val_sequences = [f'{int(seq):02d}' for seq in val_sequences]

            trainset = SemanticKitti(
                root=self.settings.data_root,
                sequences=train_sequences,
                config_path=data_config_path)

            valset = SemanticKitti(
                root=self.settings.data_root,
                sequences=val_sequences,
                config_path=data_config_path,
                has_label=val_has_label)

            self.mapped_cls_name = trainset.mapped_cls_name
            self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
            self.ignore_class = []
            for cl, weight in enumerate(self.cls_weight):
                if trainset.data_config['learning_ignore'][cl]:
                    self.cls_weight[cl] = 0
                if self.cls_weight[cl] < 1e-10:
                    self.ignore_class.append(cl)
            if self.recorder is not None:
                self.recorder.logger.info('Class weights: {}'.format(self.cls_weight))

        # nuScenes dataset
        elif self.settings.dataset == 'nuScenes':
            raise NotImplementedError("[trained_rangeformer] nuScenes dataset support is not implemented in current snippet.")
            print('----Using nuScenes dataset----')
            version = 'v1.0-mini' if self.settings.use_mini_version else 'v1.0-trainval'

            trainset = dataset.nuScenes.Nuscenes(
                dataroot=self.settings.data_root, version=version, split='train')
            valset = dataset.nuScenes.Nuscenes(
                dataroot=self.settings.data_root, version=version, split='val')

            self.mapped_cls_name = trainset.mapped_cls_name
            self.ignore_class = [0]
            self.cls_weight = np.ones((self.settings.n_classes))
            self.cls_weight[0] = 0

        else:
            raise NotImplementedError(f'Dataset {self.settings.dataset} not supported')

        # Wrap with RangeFormer loader
        train_dataset = RangeFormerLoader(
            trainset, self.settings.config,
            is_train=True, return_uproj=False, use_kpconv=False)

        val_dataset = RangeFormerLoader(
            valset, self.settings.config,
            is_train=False,
            return_uproj=bool(self.settings.test_split and self.settings.save_eval_results),
            use_kpconv=False)

        # Create data loaders
        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.settings.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.settings.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.settings.batch_size_val,
            shuffle=False,
            num_workers=self.settings.num_workers,
            sampler=val_sampler,
            pin_memory=True)

        return train_loader, val_loader, train_sampler, val_sampler

    def _initCriterion(self):
        """Initialize loss criterion."""
        loss_config = self.settings.config.get('loss', {})
        loss_type = loss_config.get('type', 'crossentropy')
        ignore_index = loss_config.get('ignore_index', 0)
        self.loss_type = loss_type
        self.loss_ignore_index = ignore_index

        if loss_type == 'crossentropy':
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(self.cls_weight).float().cuda(),
                ignore_index=ignore_index)
        elif loss_type == 'lovasz':
            criterion = Lovasz_softmax(ignore=ignore_index)
        elif loss_type == 'focal':
            from utils.optim.focal_softmax import FocalSoftmaxLoss
            criterion = FocalSoftmaxLoss(ignore=ignore_index)
        elif loss_type == 'composite':
            self.ce_loss = nn.CrossEntropyLoss(
                weight=torch.from_numpy(self.cls_weight).float().cuda(),
                ignore_index=ignore_index)
            self.dice_loss_fn = DiceLoss(ignore_index=ignore_index)
            self.lovasz_loss_fn = Lovasz_softmax(ignore=ignore_index)
            self.boundary_loss_fn = BoundaryLoss(ignore_index=ignore_index)
            self.loss_weights = {
                'ce': loss_config.get('ce_weight', 1.0),
                'dice': loss_config.get('dice_weight', 1.0),
                'lovasz': loss_config.get('lovasz_weight', 1.0),
                'boundary': loss_config.get('boundary_weight', 1.0),
            }
            criterion = None
        else:
            raise NotImplementedError(f'Loss type {loss_type} not supported')

        print(f'Using {loss_type} loss')
        return criterion

    def _composite_loss(self, logits, labels):
        ce = self.ce_loss(logits, labels)
        dice = self.dice_loss_fn(logits, labels)
        lovasz = self.lovasz_loss_fn(torch.softmax(logits, dim=1), labels)
        boundary = self.boundary_loss_fn(logits, labels)
        return (
            self.loss_weights['ce'] * ce +
            self.loss_weights['dice'] * dice +
            self.loss_weights['lovasz'] * lovasz +
            self.loss_weights['boundary'] * boundary
        )

    def _compute_loss(self, logits, labels):
        if self.loss_type == 'composite':
            return self._composite_loss(logits, labels)
        else:
            return self.criterion(logits, labels)

    def train_one_epoch(self, epoch):
        """Train for one epoch with auxiliary loss support."""
        self.model.train()
        self.metrics.reset()

        if self.settings.distributed:
            self.train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_loss_main = 0.0
        epoch_loss_aux = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack data
            if len(batch_data) == 3:
                range_images, labels, masks = batch_data
            else:
                range_images, labels = batch_data[:2]
                masks = None

            range_images = range_images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True).long()

            # Forward pass with mixed precision
            if self.fp16_scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, aux_logits = self.model(range_images)

                    # Main loss
                    loss_main = self._compute_loss(logits, labels)

                    # Auxiliary losses
                    loss_aux = 0.0
                    for aux_logit in aux_logits:
                        loss_aux += self._compute_loss(aux_logit, labels)
                    loss_aux /= len(aux_logits)

                    # Total loss
                    loss = loss_main + self.aux_loss_weight * loss_aux

                # Backward pass
                self.optimizer.zero_grad()
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            else:
                # Regular forward pass
                logits, aux_logits = self.model(range_images)

                # Main loss
                loss_main = self._compute_loss(logits, labels)

                # Auxiliary losses
                loss_aux = 0.0
                for aux_logit in aux_logits:
                    loss_aux += self._compute_loss(aux_logit, labels)
                loss_aux /= len(aux_logits)

                # Total loss
                loss = loss_main + self.aux_loss_weight * loss_aux

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Update metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                self.metrics.addBatch(preds.cpu().numpy(), labels.cpu().numpy())

            # Track losses
            epoch_loss += loss.item()
            epoch_loss_main += loss_main.item()
            epoch_loss_aux += loss_aux.item()
            num_batches += 1

            # Logging
            if batch_idx % self.settings.log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                miou_running, _ = self.metrics.getIoU()
                acc_running, _ = self.metrics.getAcc()
                recall_running, _ = self.metrics.getRecall()
                print(f'Epoch [{epoch}/{self.settings.n_epochs}] '
                      f'Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} (Main: {loss_main.item():.4f}, Aux: {loss_aux.item():.4f}) '
                      f'LR: {current_lr:.6f} '
                      f'Acc: {acc_running.item():.4f} '
                      f'mIoU: {miou_running.item():.4f} '
                      f'Recall: {recall_running.item():.4f}')

        # Compute metrics
        avg_loss = epoch_loss / num_batches
        avg_loss_main = epoch_loss_main / num_batches
        avg_loss_aux = epoch_loss_aux / num_batches

        miou, ious = self.metrics.getIoU()
        acc_mean, _ = self.metrics.getAcc()
        acc_value = acc_mean.item() if isinstance(acc_mean, torch.Tensor) else float(acc_mean)

        print(f'\nEpoch {epoch} Training Results:')
        print(f'  Loss: {avg_loss:.4f} (Main: {avg_loss_main:.4f}, Aux: {avg_loss_aux:.4f})')
        print(f'  mIoU: {miou:.4f}')
        print(f'  Accuracy: {acc_value:.4f}')

        return {
            'loss': avg_loss,
            'loss_main': avg_loss_main,
            'loss_aux': avg_loss_aux,
            'miou': miou,
            'acc': acc_value,
            'ious': ious
        }

    def validate(self, epoch):
        """Validation with auxiliary outputs."""
        self.model.eval()
        if not self.settings.test_split:
            self.metrics.reset()

        val_loss = 0.0
        num_batches = 0
        saved_predictions = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # Unpack data
                if len(batch_data) >= 5:
                    range_images, labels, masks, proj_indices, sample_indices = batch_data
                elif len(batch_data) == 3:
                    range_images, labels, masks = batch_data
                    proj_indices = None
                    sample_indices = None
                else:
                    range_images, labels = batch_data[:2]
                    masks = None
                    proj_indices = None
                    sample_indices = None

                range_images = range_images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True).long()

                # Forward pass
                logits = self._forward_with_str_views(range_images)
                preds = logits.argmax(dim=1)

                if self.settings.test_split:
                    if self.settings.save_eval_results:
                        if proj_indices is None or sample_indices is None:
                            raise RuntimeError('Test split evaluation requires return_uproj=True to recover point indices.')
                        proj_indices = proj_indices.cpu().numpy()
                        sample_indices = sample_indices.cpu().numpy()
                        preds_np = preds.cpu().numpy()
                        for b in range(preds_np.shape[0]):
                            index = int(sample_indices[b])
                            pred_map = preds_np[b]
                            proj_idx_map = proj_indices[b]
                            self._save_test_predictions(index, pred_map, proj_idx_map)
                            saved_predictions += 1
                    continue

                # Loss (main only for validation)
                loss = self._compute_loss(logits, labels)
                val_loss += loss.item()
                num_batches += 1

                # Metrics
                self.metrics.addBatch(preds.cpu().numpy(), labels.cpu().numpy())

        if self.settings.test_split:
            msg = f'\nTest split inference completed.'
            if self.settings.save_eval_results:
                msg += f' Saved {saved_predictions} scans to {self.prediction_path}.'
            print(msg)
            return {'saved_predictions': saved_predictions} if self.settings.save_eval_results else {}

        # Compute metrics
        avg_loss = val_loss / num_batches
        miou, ious = self.metrics.getIoU()
        acc, _ = self.metrics.getAcc()

        print(f'\nEpoch {epoch} Validation Results:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  mIoU: {miou:.4f}')
        print(f'  Accuracy: {acc:.4f}')

        return {
            'loss': avg_loss,
            'miou': miou,
            'acc': acc,
            'ious': ious
        }

    def _save_test_predictions(self, index: int, pred_map: np.ndarray, proj_idx_map: np.ndarray):
        """Map 2D predictions back to points and save SemanticKITTI-formatted labels."""
        if self.prediction_path is None:
            return

        sk_dataset = self.val_loader.dataset.dataset
        pointcloud, _, _ = sk_dataset.loadDataByIndex(index)
        num_points = pointcloud.shape[0]
        default_label = self.loss_ignore_index if hasattr(self, 'loss_ignore_index') else 0
        point_preds = np.ones(num_points, dtype=np.int32) * default_label
        proj_idx_map = proj_idx_map.astype(np.int64, copy=False)
        pred_map = pred_map.astype(np.int32, copy=False)
        mask = proj_idx_map >= 0
        point_preds[proj_idx_map[mask]] = pred_map[mask]

        raw_preds = sk_dataset.class_map_lut_inv[point_preds]
        seq_id, frame_id = sk_dataset.parsePathInfoByIndex(index)
        pred_dir = os.path.join(self.prediction_path, 'sequences', seq_id, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        output_path = os.path.join(pred_dir, f'{frame_id}.label')
        raw_preds.astype(np.uint32).tofile(output_path)

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        if not tools.is_main_process():
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.settings.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.settings.config
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.recorder.checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.recorder.checkpoint_path, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')

    def train(self):
        """Main training loop."""
        print('Starting RangeFormer training...')
        best_miou = 0.0

        for epoch in range(1, self.settings.n_epochs + 1):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch}/{self.settings.n_epochs}')
            print(f'{"="*60}')

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            if epoch % self.settings.val_frequency == 0:
                val_metrics = self.validate(epoch)

                # Save checkpoint
                is_best = val_metrics['miou'] > best_miou
                if is_best:
                    best_miou = val_metrics['miou']

                if epoch % self.settings.config.get('save_frequency', 5) == 0:
                    self.save_checkpoint(epoch, val_metrics, is_best=is_best)

                # Log to tensorboard
                if self.recorder and tools.is_main_process() and self.recorder.tensorboard:
                    tb = self.recorder.tensorboard
                    tb.add_scalar('val/loss', val_metrics['loss'], epoch)
                    tb.add_scalar('val/miou', val_metrics['miou'], epoch)
                    tb.add_scalar('val/acc', val_metrics['acc'], epoch)

            # Log training metrics
            if self.recorder and tools.is_main_process() and self.recorder.tensorboard:
                tb = self.recorder.tensorboard
                tb.add_scalar('train/loss', train_metrics['loss'], epoch)
                tb.add_scalar('train/loss_main', train_metrics['loss_main'], epoch)
                tb.add_scalar('train/loss_aux', train_metrics['loss_aux'], epoch)
                tb.add_scalar('train/miou', train_metrics['miou'], epoch)
                tb.add_scalar('train/acc', train_metrics['acc'], epoch)

        print('\nTraining completed!')
        print(f'Best validation mIoU: {best_miou:.4f}')


def create_rangeformer_model(settings):
    """Create RangeFormer model from settings."""
    config = {
        'H': settings.image_size[0],
        'W': settings.image_size[1],
        'num_classes': settings.n_classes,
        'depths': settings.config['rangeformer']['depths'],
        'stage_channels': settings.config['rangeformer']['stage_channels'],
        'heads': settings.config['rangeformer']['heads'],
        'decoder_unify_ch': settings.config['rangeformer'].get('decoder_unify_ch', 256),
        'mlp_ratio': settings.config['rangeformer'].get('mlp_ratio', 4.0),
        'sr_ratios': settings.config['rangeformer'].get('sr_ratios', [8, 4, 2, 1]),
    }

    model = create_rangeformer(config)

    print(f'\nRangeFormer Model Configuration:')
    print(f'  Input size: {settings.image_size}')
    print(f'  Number of classes: {settings.n_classes}')
    print(f'  Depths: {config["depths"]}')
    print(f'  Channels: {config["stage_channels"]}')
    print(f'  Heads: {config["heads"]}')

    stats = model.count_parameters_by_component()
    print(f'\nParameter counts:')
    for k, v in stats.items():
        print(f'  {k}: {v:,}')

    return model
