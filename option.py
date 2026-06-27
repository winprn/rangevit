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

import os
import yaml
import sys
import shutil
import utils.tools as tools
from utils.robust_eval import SUPPORTED_ROBUST_EVAL_TYPES


class Option(object):
    def __init__(self, config_path, args):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, 'r'))
        data_cfg = self.config.get('data', {})
        train_cfg = self.config.get('training', {})
        model_cfg = self.config.get('model', {})

        # General options
        self.seed = 1
        self.gpu = None
        self.rank = 0  # rank of distributed thread
        self.world_size = 1
        self.distributed = self.config.get('distributed', False)
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.num_workers = 4 # number of threads used for data loading

        # Data config
        self.dataset = data_cfg.get('dataset', self.config.get('dataset', None))
        self.n_classes = data_cfg.get('n_classes', self.config.get('n_classes', None))
        self.data_root = data_cfg.get('data_root', self.config.get('data_root', None))
        self.has_label = data_cfg.get('has_label', self.config.get('has_label', None))
        self.use_mini_version = False
        self.use_trainval = data_cfg.get('use_trainval', self.config.get('use_trainval', False))
        # Label-efficient / BALViT-style two-level skip:
        #   dataset_skip_step_org: applied at the parser level via the
        #     percentiles_split.json (one of 10 / 100 / 1000 for our three
        #     standard splits; set to 1 to disable).
        #   dataset_skip_step: applied at the DataLoader/RangeViewLoader
        #     level for additional per-epoch subsampling.
        #   repeat_factor: number of times to repeat each selected scan in
        #     an epoch (BALViT uses this to compensate for very small splits).
        self.dataset_skip_step_org = int(
            data_cfg.get('dataset_skip_step_org',
                         self.config.get('dataset_skip_step_org', 1))
        )
        self.dataset_skip_step = int(
            data_cfg.get('dataset_skip_step',
                         self.config.get('dataset_skip_step', 1))
        )
        self.repeat_factor = int(
            data_cfg.get('repeat_factor',
                         self.config.get('repeat_factor', 1))
        )
        # Convenience flag: enable the BALViT label-efficient protocol.
        self.label_efficient_enable = bool(
            data_cfg.get('label_efficient_enable',
                         self.config.get('label_efficient_enable',
                                         self.dataset_skip_step_org > 1))
        )

        # Train config
        self.val_only = False
        self.tta = 'none'
        self.val_frequency = train_cfg.get('val_frequency', self.config.get('val_frequency', 10))
        self.test_split = False
        self.n_epochs = train_cfg.get('n_epochs', self.config.get('n_epochs', None))  # number of total epochs
        self.batch_size = train_cfg.get('batch_size', self.config.get('batch_size', None))  # mini-batch size
        self.batch_size_val = train_cfg.get('batch_size_val', self.config.get('batch_size_val', 1)) # validation batch size
        self.lr = train_cfg.get('lr', self.config.get('lr', None))
        self.min_lr = float(train_cfg.get('min_lr', self.config.get('min_lr', 0.0)))
        self.warmup_epochs = train_cfg.get('warmup_epochs', self.config.get('warmup_epochs', 10))
        loss_cfg = train_cfg.get('loss', self.config.get('loss', {}))
        focal_cfg = loss_cfg.get('focal_loss', {})
        boundary_cfg = loss_cfg.get('boundary_loss', {})
        lovasz_cfg = loss_cfg.get('lovasz_loss', {})
        aux_cfg = loss_cfg.get('aux_loss', {})

        focal_type_raw = str(focal_cfg.get('type', self.config.get('focal_loss_type', 'focal'))).lower()
        # Accept alias from configs: "class_weight_focal" -> "class_weighted_focal"
        if focal_type_raw == 'class_weight_focal':
            focal_type_raw = 'class_weighted_focal'
        self.focal_loss_type = focal_type_raw
        self.focal_gamma = float(focal_cfg.get('gamma', self.config.get('focal_gamma', 2.0)))
        self.focal_ignore_index = int(focal_cfg.get('ignore_index', self.config.get('focal_ignore_index', 0)))
        self.focal_loss_weight = float(focal_cfg.get('weight', self.config.get('focal_loss_weight', 1.0)))
        self.lovasz_loss_weight = float(lovasz_cfg.get('weight', self.config.get('lovasz_loss_weight', 1.0)))
        self.boundary_loss_weight = float(boundary_cfg.get('weight', self.config.get('boundary_loss_weight', 0.0)))
        self.class_weights = train_cfg.get('class_weights', self.config.get('class_weights', None))
        self.log_frequency = 100
        self.train_result_frequency = train_cfg.get('train_result_frequency', self.config.get('train_result_frequency', 100))
        self.use_fp16 = train_cfg.get('use_fp16', self.config.get('use_fp16', False)) # for mixed-precision training
        self.aux_loss_weight = float(aux_cfg.get('weight', self.config.get('aux_loss_weight', 0.3)))
        save_epochs_raw = train_cfg.get('save_epochs_at', self.config.get('save_epochs_at', []))
        if self.dataset is None:
            raise ValueError("Missing required config: data.dataset (or legacy top-level dataset)")
        if self.n_classes is None:
            raise ValueError("Missing required config: data.n_classes (or legacy top-level n_classes)")
        if self.has_label is None:
            raise ValueError("Missing required config: data.has_label (or legacy top-level has_label)")
        if self.n_epochs is None:
            raise ValueError("Missing required config: training.n_epochs (or legacy top-level n_epochs)")
        if self.batch_size is None:
            raise ValueError("Missing required config: training.batch_size (or legacy top-level batch_size)")
        if self.lr is None:
            raise ValueError("Missing required config: training.lr (or legacy top-level lr)")
        if save_epochs_raw is None:
            save_epochs_raw = []
        if not isinstance(save_epochs_raw, (list, tuple)):
            raise ValueError('save_epochs_at must be a list of 1-based epoch indices.')
        save_epochs_cleaned = []
        for e in save_epochs_raw:
            if not isinstance(e, int):
                raise ValueError('All save_epochs_at entries must be integers.')
            if e <= 0:
                raise ValueError('save_epochs_at entries must be positive (1-based).')
            if e >= self.n_epochs:
                raise ValueError(f'save_epochs_at entry {e} must be < total epochs ({self.n_epochs}).')
            save_epochs_cleaned.append(e)
        self.save_epochs_at = sorted(set(save_epochs_cleaned))
        if self.min_lr < 0.0:
            raise ValueError('min_lr must be >= 0.')
        if self.min_lr > self.lr:
            raise ValueError(f'min_lr ({self.min_lr}) must be <= lr ({self.lr}).')
        if self.focal_loss_type not in ('focal', 'class_weighted_focal'):
            raise ValueError("focal_loss_type must be one of: 'focal', 'class_weighted_focal'")
        if self.focal_gamma < 0.0:
            raise ValueError('focal_gamma must be >= 0.')
        if self.focal_loss_weight < 0.0:
            raise ValueError('focal_loss.weight must be >= 0.')
        if self.lovasz_loss_weight < 0.0:
            raise ValueError('lovasz_loss.weight must be >= 0.')
        if self.boundary_loss_weight < 0.0:
            raise ValueError('boundary_loss.weight must be >= 0.')
        if self.aux_loss_weight < 0.0:
            raise ValueError('aux_loss.weight must be >= 0.')
        if self.focal_ignore_index < 0 or self.focal_ignore_index >= self.n_classes:
            raise ValueError(f'focal_ignore_index must be in [0, n_classes-1], got {self.focal_ignore_index}')
        if self.class_weights is not None:
            if not isinstance(self.class_weights, (list, tuple)):
                raise ValueError('class_weights must be a list/tuple when provided')
            if len(self.class_weights) != self.n_classes:
                raise ValueError(f'class_weights length ({len(self.class_weights)}) must equal n_classes ({self.n_classes})')
            self.class_weights = [float(w) for w in self.class_weights]
        if self.label_efficient_enable:
            if self.dataset not in ('SemanticKitti', 'nuScenes'):
                raise ValueError('Label-efficient training is currently supported only for SemanticKitti and nuScenes.')
            if self.dataset == 'SemanticKitti':
                if self.dataset_skip_step_org not in (10, 100, 1000):
                    raise ValueError(
                        'data.dataset_skip_step_org must be one of (10, 100, 1000) '
                        'when label-efficient training is enabled for SemanticKitti.'
                    )
            elif self.dataset == 'nuScenes':
                if self.dataset_skip_step not in (10, 100, 1000):
                    raise ValueError(
                        'data.dataset_skip_step must be one of (10, 100, 1000) '
                        'when label-efficient training is enabled for nuScenes.'
                    )
            if self.dataset_skip_step < 1:
                raise ValueError('data.dataset_skip_step must be a positive integer.')
            if self.repeat_factor < 1:
                raise ValueError('data.repeat_factor must be a positive integer.')



        # Model config
        self.vit_backbone = model_cfg.get('vit_backbone', self.config.get('vit_backbone', 'vit_small_patch16_384'))
        self.in_channels = model_cfg.get('in_channels', self.config.get('in_channels', 5))
        self.patch_size = model_cfg.get('patch_size', self.config.get('patch_size', [2, 8]))
        self.patch_stride = model_cfg.get('patch_stride', self.config.get('patch_stride', [2, 8]))
        self.image_size = model_cfg.get('image_size', self.config.get('image_size', [32, 384]))
        self.window_size = model_cfg.get('window_size', self.config.get('window_size', [32, 384]))
        self.window_stride = model_cfg.get('window_stride', self.config.get('window_stride', [32, 256]))
        self.original_image_size = model_cfg.get('original_image_size', self.config.get('original_image_size', [32, 2048]))
        # Full-image mode: set train_full_image=True to disable training crops,
        # and use_sliding_window=False to run full-frame inference/validation.
        self.train_full_image = model_cfg.get('train_full_image', self.config.get('train_full_image', False))
        self.use_sliding_window = model_cfg.get('use_sliding_window', self.config.get('use_sliding_window', True))

        # Freeze encoder params
        self.freeze_vit_encoder = self.config.get('freeze_vit_encoder', False)
        self.unfreeze_layernorm = self.config.get('unfreeze_layernorm', False)
        self.unfreeze_attn = self.config.get('unfreeze_attn', False)
        self.unfreeze_ffn = self.config.get('unfreeze_ffn', False)

        # Stem
        self.conv_stem = self.config.get('conv_stem', 'ConvStem')
        self.stem_base_channels = self.config.get('stem_base_channels', 32)
        self.D_h = self.config.get('D_h', 256)

        # Decoder
        # Backward compatible formats:
        # 1) legacy flat: decoder: "fpn", fuse_* at root
        # 2) nested: decoder: { name: "fpn", tinyvim_fuse_aux: {...} }
        decoder_cfg = self.config.get('decoder', 'up_conv')
        if isinstance(decoder_cfg, dict):
            self.decoder = decoder_cfg.get('name', 'up_conv')
            tinyvim_fuse_cfg = decoder_cfg.get('tinyvim_fuse_aux', {})
        else:
            self.decoder = decoder_cfg
            tinyvim_fuse_cfg = {}

        self.skip_filters = self.config.get('skip_filters', 0)
        self.fuse_proj_channels = int(
            tinyvim_fuse_cfg.get('proj_channels', self.config.get('fuse_proj_channels', 128))
        )
        self.fuse_mid_channels = int(
            tinyvim_fuse_cfg.get('mid_channels', self.config.get('fuse_mid_channels', 256))
        )
        self.fuse_out_channels = int(
            tinyvim_fuse_cfg.get('out_channels', self.config.get('fuse_out_channels', 128))
        )
        self.fuse_preproj = bool(
            tinyvim_fuse_cfg.get('preproj', self.config.get('fuse_preproj', True))
        )
        self.aux_enable = bool(
            tinyvim_fuse_cfg.get('aux_enable', self.config.get('aux_enable', True))
        )
        # aux_loss_weight is configured in the Train config via loss.aux_loss.weight.

        # 3D refiner / post-processing
        self.use_kpconv = False
        self.use_knn = False
        self.knn_search = self.config.get('knn_search', 7)
        self.knn_k = self.config.get('knn_k', 5)
        self.knn_sigma = self.config.get('knn_sigma', 1.0)
        self.knn_cutoff = self.config.get('knn_cutoff', 1.0)

        point_postproc = str(self.config.get('point_postproc', 'none')).lower()
        if point_postproc == 'kpconv':
            self.use_kpconv = True
            self.use_knn = False
        elif point_postproc == 'knn':
            self.use_kpconv = False
            self.use_knn = True
        elif point_postproc in ('none', 'off', 'false', '0'):
            self.use_kpconv = False
            self.use_knn = False
        else:
            raise ValueError('point_postproc must be one of: kpconv, knn, none')

        # Range image-level augmentation
        self.range_aug = self.config.get('range_aug', False)

        # Validation-time robustness sensitivity analysis.
        robust_eval_cfg = self.config.get('robust_eval', {})
        self.robust_eval_enabled = bool(robust_eval_cfg.get('enabled', False))
        self.robust_eval_type = str(robust_eval_cfg.get('type', 'none')).lower()
        self.robust_eval_severity = float(robust_eval_cfg.get('severity', 0.0))
        self.robust_eval_seed = int(robust_eval_cfg.get('seed', 42))

        # Checkpoint model
        self.checkpoint = self.config.get('checkpoint', None)
        self.pretrained_model = self.config.get('pretrained_model', None)
        self.finetune_pretrained_model = self.config.get('finetune_pretrained_model', False)

        # Loading pre-trained patch and positional embeddings
        self.reuse_pos_emb = self.config.get('reuse_pos_emb', False)
        self.reuse_patch_emb = self.config.get('reuse_patch_emb', False)

        # Channel adaptation method for pretrained weights (RGB -> LiDAR)
        self.pretrained_channel_adaptation = self.config.get('pretrained_channel_adaptation', 'repeat')


        # Save results
        self.id = self.config['id'] # name to identify the run
        self.save_eval_results = False

        save_root = args.save_path if args.save_path is not None else self.config.get('save_path', None)
        if save_root is None:
            raise ValueError('save_path must be provided either via config file or command line.')
        self.save_path = os.path.join(save_root, 'log_{}'.format(self.id))

        # MLflow config
        mlflow_cfg = self.config.get('mlflow', {})
        self.mlflow_enable = mlflow_cfg.get('enable', False)
        self.mlflow_tracking_uri = mlflow_cfg.get('tracking_uri', None)
        self.mlflow_experiment_name = mlflow_cfg.get(
            'experiment_name', 'RangeViT')
        self.mlflow_run_name = mlflow_cfg.get('run_name', self.id)
        self.mlflow_nested = mlflow_cfg.get('nested', False)
        tags_cfg = mlflow_cfg.get('tags', {})
        self.mlflow_tags = tags_cfg if isinstance(tags_cfg, dict) else {}
        self.mlflow_description = mlflow_cfg.get('description', None)
        self.mlflow_log_checkpoints = mlflow_cfg.get('log_checkpoints', True)
        self.mlflow_log_code_snapshot = mlflow_cfg.get(
            'log_code_snapshot', True)

        # -----------------------------------------------------
        # Check options

        # There is no skip connection if no convolutional stem is used or the linear decoder is used.
        # (If no convolutional stem is used, then we use PatchEmbedding istead).
        if self.conv_stem == 'none' or self.decoder == 'linear':
            assert self.skip_filters == 0

        # If there is a skip connection, it's channel dim has to be D_h.
        if self.skip_filters > 0:
            assert self.skip_filters == self.D_h

        # If a convolutional stem is used, patch_size = patch_stride and there is no patch embedding
        # so we can't load pre-trained weights in the patch embeddings.
        if self.conv_stem != 'none':
            assert self.patch_size == self.patch_stride
            assert self.reuse_patch_emb == False

        # When using the KPConv layer, the decoder has to be up_conv.
        if self.use_kpconv:
            assert self.decoder in ('up_conv', 'fpn', 'fpn_gated', 'fpn_gated_detail',
                                    'fpn_residual', 'fpn_cross_attn', 'fpn_residual_cross_attn'), \
                   'KPConv supported only with up_conv, fpn, fpn_gated, or fpn_gated_detail decoders'
        if self.decoder == 'fpn_residual':
            assert self.vit_backbone.startswith('tinyvim'), 'fpn_residual decoder requires a TinyViM backbone'
            assert self.skip_filters == 0, 'fpn_residual decoder does not use skip_filters'
        if self.decoder == 'fpn_cross_attn':
            assert self.vit_backbone.startswith('tinyvim'), 'fpn_cross_attn decoder requires a TinyViM backbone'
            assert self.skip_filters == 0, 'fpn_cross_attn decoder does not use skip_filters'
        if self.decoder == 'fpn_residual_cross_attn':
            assert self.vit_backbone.startswith('tinyvim'), 'fpn_residual_cross_attn decoder requires a TinyViM backbone'
            assert self.skip_filters == 0, 'fpn_residual_cross_attn decoder does not use skip_filters'
        if self.decoder == 'fpn_gated':
            assert self.vit_backbone.startswith('tinyvim'), 'fpn_gated decoder requires a TinyViM backbone'
            assert self.skip_filters == 0, 'fpn_gated decoder does not use skip_filters'
        if self.decoder == 'fpn_gated_detail':
            assert self.vit_backbone.startswith('tinyvim'), 'fpn_gated_detail decoder requires a TinyViM backbone'
            assert self.skip_filters == 0, 'fpn_gated_detail decoder does not use skip_filters'
        if self.decoder == 'tinyvim_fuse_aux':
            assert self.vit_backbone.startswith('tinyvim'), 'tinyvim_fuse_aux decoder requires a TinyViM backbone'
            assert self.use_kpconv is False, 'tinyvim_fuse_aux decoder does not support KPConv'
            assert self.skip_filters == 0, 'tinyvim_fuse_aux decoder does not use skip_filters'
            assert self.aux_loss_weight >= 0.0, 'aux_loss_weight must be >= 0'
        if self.use_kpconv and self.use_knn:
            raise AssertionError('use_kpconv and use_knn cannot both be True')
        if self.robust_eval_type not in SUPPORTED_ROBUST_EVAL_TYPES:
            raise ValueError(
                f"robust_eval.type must be one of: {', '.join(sorted(SUPPORTED_ROBUST_EVAL_TYPES))}"
            )
        if self.robust_eval_severity < 0.0 or self.robust_eval_severity > 1.0:
            raise ValueError('robust_eval.severity must be in [0, 1].')
        if self.robust_eval_enabled and self.use_kpconv:
            raise AssertionError('robust_eval is supported only for the non-KPConv validation path')

        # The following hyperparameters have to be tuples or lists with two elements.
        tuple_list = [self.patch_size, self.patch_stride,
                      self.image_size, self.window_size, self.window_stride,
                      self.original_image_size]
        for i in tuple_list:
            assert isinstance(i, (list, tuple))
            assert len(i) == 2

        if self.train_full_image:
            # Full-image mode expects the projected range map to be at least as large as original_image_size
            proj_h = self.config['sensor']['proj_h']
            proj_w = self.config['sensor']['proj_w']
            assert self.original_image_size[0] <= proj_h and self.original_image_size[1] <= proj_w, \
                f"original_image_size {self.original_image_size} must fit inside sensor projection {(proj_h, proj_w)} for full-image mode"
            if self.image_size != self.original_image_size:
                print(f"[RangeViT] train_full_image=True overrides random crop; using original_image_size {self.original_image_size}")

        # No patch and positional embeddings are loaded when training from scratch.
        if self.pretrained_model == None:
            assert self.reuse_patch_emb == self.reuse_pos_emb == False

        # Validate channel adaptation method
        if self.pretrained_channel_adaptation not in ('repeat', 'grayscale'):
            raise ValueError(f"pretrained_channel_adaptation must be 'repeat' or 'grayscale', "
                           f"got '{self.pretrained_channel_adaptation}'")


    def check_path(self):
        if tools.is_main_process():
            if os.path.exists(self.save_path):
                print('WARNING: Directory exist: {}'.format(self.save_path))

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
