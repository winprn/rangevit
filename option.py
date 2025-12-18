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
        self.num_workers = 4 # number of threads used for data loading

        # Data config
        self.dataset = self.config['dataset']
        self.n_classes = self.config['n_classes']
        self.data_root = self.config.get('data_root', None)
        self.has_label = self.config['has_label']
        self.use_mini_version = False
        self.use_trainval = self.config.get('use_trainval', False)

        # Train config
        self.val_only = False
        self.val_frequency = self.config.get('val_frequency', 10)
        self.test_split = False
        self.n_epochs = self.config['n_epochs']  # number of total epochs
        self.batch_size = self.config['batch_size']  # mini-batch size
        self.batch_size_val = self.config.get('batch_size_val', 1) # validation batch size
        self.lr = self.config['lr']
        self.warmup_epochs = self.config.get('warmup_epochs', 10)
        self.log_frequency = 100
        self.train_result_frequency = self.config.get('train_result_frequency', 100)
        self.use_fp16 = self.config.get('use_fp16', False) # for mixed-precision training


        # Model config
        self.vit_backbone = self.config.get('vit_backbone', 'vit_small_patch16_384')
        self.in_channels = self.config.get('in_channels', 5)
        self.patch_size = self.config.get('patch_size', [2, 8])
        self.patch_stride = self.config.get('patch_stride', [2, 8])
        self.image_size = self.config.get('image_size', [32, 384])
        self.window_size = self.config.get('window_size', [32, 384])
        self.window_stride = self.config.get('window_stride', [32, 256])
        self.original_image_size = self.config.get('original_image_size', [32, 2048])

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

        # 3D refiner / post-processing
        self.use_kpconv = self.config.get('use_kpconv', True)
        self.use_knn = self.config.get('use_knn', False)
        self.knn_search = self.config.get('knn_search', 13)
        self.knn_k = self.config.get('knn_k', 5)
        self.knn_sigma = self.config.get('knn_sigma', 1.0)
        self.knn_cutoff = self.config.get('knn_cutoff', 1.0)

        point_postproc = self.config.get('point_postproc', None)
        if point_postproc is not None:
            point_postproc = str(point_postproc).lower()
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


        # Checkpoint model
        self.checkpoint = self.config.get('checkpoint', None)
        self.pretrained_model = self.config.get('pretrained_model', None)
        self.finetune_pretrained_model = self.config.get('finetune_pretrained_model', False)

        # Loading pre-trained patch and positional embeddings
        self.reuse_pos_emb = self.config.get('reuse_pos_emb', False)
        self.reuse_patch_emb = self.config.get('reuse_patch_emb', False)


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
            assert self.decoder == 'up_conv'
        if self.use_kpconv and self.use_knn:
            raise AssertionError('use_kpconv and use_knn cannot both be True')

        # The following hyperparameters have to be tuples or lists with two elements.
        tuple_list = [self.patch_size, self.patch_stride,
                      self.image_size, self.window_size, self.window_stride,
                      self.original_image_size]
        for i in tuple_list:
            assert isinstance(i, (list, tuple))
            assert len(i) == 2

        # No patch and positional embeddings are loaded when training from scratch.
        if self.pretrained_model == None:
            assert self.reuse_patch_emb == self.reuse_pos_emb == False


    def check_path(self):
        if tools.is_main_process():
            if os.path.exists(self.save_path):
                print('WARNING: Directory exist: {}'.format(self.save_path))

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
