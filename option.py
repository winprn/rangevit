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


class Option(object):
    def __init__(self, config_path, args):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, 'r'))

        # General options
        self.seed = 1
        self.gpu = None
        self.rank = 0  # rank of distributed thread
        self.world_size = 1
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.num_workers = self.config.get('num_workers', 4)

        # Data config
        dataset_config = self.config.get('dataset', {})
        if isinstance(dataset_config, str):
            self.dataset = dataset_config
            dataset_n_classes = None
        else:
            self.dataset = dataset_config.get('name', 'SemanticKitti')  # Extract just the name
            dataset_n_classes = dataset_config.get('n_classes', None)
        self.n_classes = self.config.get('n_classes', dataset_n_classes)
        if self.n_classes is None:
            raise KeyError("n_classes must be defined in config or dataset section")
        self.data_root = args.data_root
        self.has_label = self.config.get('has_label', True)
        self.use_mini_version = False
        self.use_trainval = self.config.get('use_trainval', False)

        # Train config
        self.val_only = False
        self.val_frequency = self.config.get('val_frequency', 10)
        self.test_split = False
        self.n_epochs = self.config.get('n_epochs', 150)
        self.batch_size = self.config.get('batch_size', 8)
        self.batch_size_val = self.config.get('batch_size_val', 1)
        self.lr = self.config.get('lr', 0.0006)
        self.warmup_epochs = self.config.get('warmup_epochs', 10)
        self.log_frequency = 100
        self.train_result_frequency = self.config.get('train_result_frequency', 100)
        self.use_fp16 = self.config.get('use_fp16', False)

        # Model config
        self.vit_backbone = self.config.get('vit_backbone', 'swin_small_patch4_window7_224')
        self.in_channels = self.config.get('in_channels', 5)
        self.patch_size = self.config.get('patch_size', [4, 8])
        self.patch_stride = self.config.get('patch_stride', [4, 8])
        self.image_size = self.config.get('image_size', [64, 384])
        self.window_size = self.config.get('window_size', 7)
        self.window_stride = self.config.get('window_stride', [64, 256])
        self.original_image_size = self.config.get('original_image_size', [64, 2048])

        # Convert window_size to list for Swin backbones
        if isinstance(self.window_size, int) and self.vit_backbone.startswith('swin'):
            self.window_size = [self.window_size, self.window_size]

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
        self.decoder = self.config.get('decoder', 'up_conv')
        self.skip_filters = self.config.get('skip_filters', 0)
        self.up_conv_d_decoder = self.config.get('up_conv_d_decoder', 256)
        self.up_conv_scale_factor = self.config.get('up_conv_scale_factor', [4, 8])

        # 3D refiner
        self.use_kpconv = self.config.get('use_kpconv', True)

        # Checkpoint model
        self.checkpoint = self.config.get('checkpoint', None)
        self.pretrained_model = self.config.get('pretrained_model', None)
        self.finetune_pretrained_model = self.config.get('finetune_pretrained_model', False)

        # Loading pre-trained patch and positional embeddings
        self.reuse_pos_emb = self.config.get('reuse_pos_emb', False)
        self.reuse_patch_emb = self.config.get('reuse_patch_emb', False)

        # Save results
        self.id = self.config.get('id', 'experiment')
        self.save_eval_results = False
        self.save_path = os.path.join(args.save_path, 'log_{}'.format(self.id))

        # Check options
        if self.conv_stem == 'none' or self.decoder == 'linear':
            assert self.skip_filters == 0
        if self.skip_filters > 0:
            assert self.skip_filters == self.D_h
        if self.conv_stem != 'none':
            assert self.patch_size == self.patch_stride
            assert self.reuse_patch_emb == False
        if self.use_kpconv:
            assert self.decoder == 'up_conv'
        if self.pretrained_model is None:
            assert self.reuse_patch_emb == self.reuse_pos_emb == False

        # Validate list/tuple parameters
        tuple_list = {
            'patch_size': self.patch_size,
            'patch_stride': self.patch_stride,
            'image_size': self.image_size,
            'window_size': self.window_size,
            'window_stride': self.window_stride,
            'original_image_size': self.original_image_size,
            'up_conv_scale_factor': self.up_conv_scale_factor
        }
        for param_name, param_value in tuple_list.items():
            if not isinstance(param_value, (list, tuple)):
                raise ValueError(f"{param_name} must be a list or tuple, got {type(param_value)}: {param_value}")
            if len(param_value) != 2:
                raise ValueError(f"{param_name} must have exactly 2 elements, got {len(param_value)}: {param_value}")
            for elem in param_value:
                if not isinstance(elem, (int, float)):
                    raise ValueError(f"Elements of {param_name} must be int or float, got {type(elem)}: {elem}")

        # Set n_layers and n_heads for Swin
        self.depths = self.config.get('depths', [2, 2, 18, 2])
        self.num_heads = self.config.get('num_heads', [3, 6, 12, 24])
        self.mlp_ratio = self.config.get('mlp_ratio', 4.0)
        self.d_model = self.config.get('d_model', 96)
        self.dropout = self.config.get('dropout', 0.0)
        self.drop_path_rate = self.config.get('drop_path_rate', 0.0)
        if self.vit_backbone.startswith('swin'):
            self.n_layers = sum(self.depths)
        else:
            self.n_layers = self.config.get('n_layers', 12)
        self.n_heads = self.config.get('n_heads', 6)

    def check_path(self):
        if tools.is_main_process():
            if os.path.exists(self.save_path):
                print('WARNING: Directory exist: {}'.format(self.save_path))
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)