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
import argparse
import os
import sys
import datetime
import time
import numpy as np
from contextlib import nullcontext
from typing import Dict, List, Optional

from option import Option
from train import Trainer
import models
import utils
import utils.tools as tools
from models.model_utils import resize_pos_embed
from utils import mlflow_utils
from utils.discord import notify_run_completion, post_message


def build_rangevit_model(settings, pretrained_path=None):
    print('==> Building RangeViT model ...')
    print(f"settings.in_channels = {settings.in_channels}")
    print(f"settings.vit_backbone = {settings.vit_backbone}")
    print(f"settings.image_size = {settings.image_size}")
    print(f"settings.patch_size = {settings.patch_size}")
    print(f"settings.patch_stride = {settings.patch_stride}")
    print(f"settings.reuse_pos_emb = {settings.reuse_pos_emb}")
    print(f"settings.reuse_patch_emb = {settings.reuse_patch_emb}")
    print(f"settings.conv_stem = {settings.conv_stem}")
    print(f"settings.stem_base_channels = {settings.stem_base_channels}")
    print(f"settings.D_h = {settings.D_h}")
    print(f"settings.skip_filters = {settings.skip_filters}")
    print(f"settings.decoder = {settings.decoder}")
    print(f"settings.use_kpconv = {settings.use_kpconv}")
    print(f"pretrained_path = {pretrained_path}")
    model = models.RangeViT(
        in_channels=settings.in_channels,
        n_cls=settings.n_classes,
        backbone=settings.vit_backbone,
        image_size=settings.image_size,
        pretrained_path=pretrained_path,
        new_patch_size=settings.patch_size,
        new_patch_stride=settings.patch_stride,
        reuse_pos_emb=settings.reuse_pos_emb,
        reuse_patch_emb=settings.reuse_patch_emb,
        conv_stem=settings.conv_stem,
        stem_base_channels=settings.stem_base_channels,
        stem_hidden_dim=settings.D_h,
        skip_filters=settings.skip_filters,
        decoder=settings.decoder,
        up_conv_d_decoder=settings.D_h,
        up_conv_scale_factor=settings.patch_stride,
        use_kpconv=settings.use_kpconv)
    return model


def _build_run_context_lines(args: argparse.Namespace, settings: Option) -> List[str]:
    """
    Gather useful runtime metadata for Discord notifications.
    """
    lines: List[str] = [
        f"- Run ID: {settings.id}",
        f"- Config: {args.config_path}",
        f"- Data root: {settings.data_root}",
        f"- Save path: {settings.save_path}",
    ]

    if settings.checkpoint:
        lines.append(f"- Checkpoint: {settings.checkpoint}")
    elif settings.pretrained_model:
        lines.append(f"- Pretrained model: {settings.pretrained_model}")

    if settings.val_only:
        lines.append("- Mode: validation-only")
    if settings.test_split:
        lines.append("- Split: test")

    launch_command = " ".join(sys.argv)
    if launch_command:
        lines.append(f"- Command: {launch_command}")

    screen_id = os.environ.get("SCREEN") or os.environ.get("STY") or os.environ.get("TMUX")
    if not screen_id:
        raise RuntimeError(
            "Screen/TMUX session not detected. "
            "If you're running manually (not inside screen/tmux), you can temporarily set one of the env vars:\n"
            "  export SCREEN=1    # (Linux/macOS Bash)\n"
            "  set SCREEN=1       # (Windows CMD)\n"
            "  $env:SCREEN = 1    # (Windows PowerShell)"
        )

    lines.append(f"- Screen ID: {screen_id}")

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        lines.append(f"- CUDA_VISIBLE_DEVICES: {cuda_devices}")

    world_size = os.environ.get("WORLD_SIZE")
    if world_size:
        lines.append(f"- WORLD_SIZE: {world_size}")

    return lines


def _is_notification_master() -> bool:
    """
    Determine whether the current process should send Discord notifications.
    Handles the pre-distributed initialization case by checking the RANK env var.
    """
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            return rank_env == "0"
    return tools.is_main_process()


def _prepare_settings(args: argparse.Namespace) -> Option:
    """
    Instantiate an Option object and normalize settings according to CLI args.
    Shared by training and follow-up evaluation flows.
    """
    settings = Option(args.config_path, args)

    settings.id = args.id if args.id is not None else settings.id
    settings.pretrained_model = args.pretrained_model if args.pretrained_model is not None else settings.pretrained_model
    if args.checkpoint is not None:
        settings.checkpoint = args.checkpoint
        settings.pretrained_model = None
        settings.finetune_pretrained_model = False

    if args.val_only and args.window_stride is not None:
        settings.window_stride = [settings.window_stride[0], args.window_stride]
        print(f'WINDOW STRIDE: {settings.window_stride}')

    settings.data_root = args.data_root
    settings.use_mini_version = args.mini
    settings.val_only = args.val_only
    settings.test_split = args.test_split
    settings.save_eval_results = args.save_eval_results
    settings.log_frequency = args.log_frequency
    settings.num_workers = args.num_workers
    settings.seed = args.seed

    # No patch and positional embeddings are loaded when training from scratch.
    if settings.pretrained_model is None:
        settings.reuse_patch_emb = False
        settings.reuse_pos_emb = False

    if settings.val_only:
        settings.save_path = os.path.join(settings.save_path, f'Eval_{settings.id}')

    return settings


class Experiment(object):
    def __init__(self, settings: Option, mlflow_active: bool = False):
        self.settings = settings
        self.mlflow_active = mlflow_active

        # Init gpu

        if tools.is_dist_avail_and_initialized():
            tools.init_distributed_mode(self.settings)
        # torch.distributed.barrier()

        self.settings.check_path()

        # Set random seed
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.settings.seed)
            torch.cuda.set_device(self.settings.gpu)
            torch.backends.cudnn.benchmark = True
        np.random.seed(self.settings.seed)
        # torch.cuda.set_device(self.settings.gpu)
        # torch.cuda.set_device("cuda:0")

        torch.backends.cudnn.benchmark = True

        # Init checkpoint
        self.recorder = None
        if tools.is_main_process():
            self.recorder = utils.tools.Recorder(self.settings, self.settings.save_path)

        self.prediction_path = os.path.join(self.settings.save_path, 'preds')

        self.epoch_start = 0

        # Init model
        self.model = self._initModel()

        # Init trainer
        self.trainer = Trainer(
            self.settings,
            self.model,
            self.recorder,
            mlflow_step_logger=self._log_step_metrics if self.mlflow_active else None,
        )

        # Load checkpoint
        self._loadCheckpoint()

    def _log_metrics(self, metrics, mode: str, epoch: int):
        if not self.mlflow_active or metrics is None:
            return
        for key, value in metrics.items():
            try:
                mlflow_utils.log_metric(f'{mode.lower()}_{key.lower()}', float(value), step=epoch)
            except Exception:
                continue

    def _log_step_metrics(self, mode: str, epoch: int, step_index: int, metrics: Dict[str, float]):
        if not self.mlflow_active:
            return
        prefix = mode.lower()
        for key, value in metrics.items():
            try:
                mlflow_utils.log_metric(f'{prefix}_{key.lower()}', float(value), step=step_index)
            except Exception:
                continue

    def _log_best_metric(self, metric_name: str, value: float):
        if not self.mlflow_active:
            return
        try:
            mlflow_utils.log_metric(f'best_val_{metric_name.lower()}', float(value))
        except Exception:
            pass

    def _log_training_time(self, total_seconds: float):
        if not self.mlflow_active:
            return
        try:
            mlflow_utils.log_metric('total_training_time_sec', float(total_seconds))
        except Exception:
            pass


    def _initModel(self):
        # Model
        model = build_rangevit_model(
            self.settings,
            pretrained_path=self.settings.pretrained_model)

        # Freezing the ViT encoder weights.
        if self.settings.freeze_vit_encoder:
            print('==> Freeze the ViT encoder (without the pos_embed and stem)')
            for param in model.rangevit.encoder.blocks.parameters():
                param.requires_grad = False

            model.rangevit.encoder.norm.weight.requires_grad = False
            model.rangevit.encoder.norm.bias.requires_grad = False

            # Unfreeze the LayerNorm layers
            if self.settings.unfreeze_layernorm:
                print('==> Unfreeze the LN layers')
                model.rangevit.encoder.norm.weight.requires_grad = True
                model.rangevit.encoder.norm.bias.requires_grad = True
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].norm1.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm1.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm2.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].norm2.bias.requires_grad = True

            if self.settings.unfreeze_attn:
                print('==> Unfreeze the ATTN layers: qkv and proj')
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].attn.qkv.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.qkv.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.proj.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].attn.proj.bias.requires_grad = True

            if self.settings.unfreeze_ffn:
                print('==> Unfreeze the FFN layers: mlp.fc1 and mlp.fc2')
                for block_id in range(0, len(model.rangevit.encoder.blocks)):
                    model.rangevit.encoder.blocks[block_id].mlp.fc1.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc1.bias.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc2.weight.requires_grad = True
                    model.rangevit.encoder.blocks[block_id].mlp.fc2.bias.requires_grad = True


        if self.recorder is not None:
            # Print the model architecture
            # self.recorder.logger.info(f'model = {model}')

            # Count the number of model parameters
            stats = model.counter_model_parameters()
            if hasattr(model, 'counter_model_parameters'):
                self.recorder.logger.info(f'Number of model parameters:')
                for key, val in stats.items():
                    self.recorder.logger.info(f'==> {key}: {val}')

        return model


    def _loadCheckpoint(self):
        if self.settings.checkpoint is not None:
            print(f'Resume training from checkpoint {self.settings.checkpoint}')
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError('checkpoint file not found: {}'.format(self.settings.checkpoint))

            checkpoint_data = torch.load(self.settings.checkpoint, map_location='cpu')

            if self.settings.finetune_pretrained_model:
                # When fine-tuning a segmentation model previously pre-trained to another dataset then it
                # is necessary to adapt the (a) pos_embeds and (b) to remove the classification head.
                image_size = self.model.rangevit.encoder.image_size
                patch_stride = self.model.rangevit.encoder.patch_stride
                if (self.model.rangevit.encoder.pos_embed.shape != checkpoint_data['model']['rangevit.encoder.pos_embed'].shape):
                    assert self.model.rangevit.encoder.pos_embed.shape[2] == checkpoint_data['model']['rangevit.encoder.pos_embed'].shape[2]
                    gs_new_h = int(image_size[0] // patch_stride[0])
                    gs_new_w = int(image_size[1] // patch_stride[1])
                    num_extra_tokens = 1
                    assert (gs_new_h * gs_new_w + num_extra_tokens) == self.model.rangevit.encoder.pos_embed.shape[1]
                    old_len = checkpoint_data['model']['rangevit.encoder.pos_embed'].shape[1] - num_extra_tokens # remove one for the classification token

                    gs_old_w = gs_new_w
                    gs_old_h = old_len // gs_old_w
                    checkpoint_data['model']['rangevit.encoder.pos_embed'] = (
                        resize_pos_embed(checkpoint_data['model']['rangevit.encoder.pos_embed'],
                                         grid_old_shape=(gs_old_h, gs_old_w),
                                         grid_new_shape=(gs_new_h, gs_new_w),
                                         num_extra_tokens=num_extra_tokens))
                assert self.model.rangevit.encoder.pos_embed.shape == checkpoint_data['model']['rangevit.encoder.pos_embed'].shape

                for key in ('rangevit.kpclassifier.head.weight', 'rangevit.kpclassifier.head.bias'):
                    del checkpoint_data['model'][key]

            checkpoint_data_model = checkpoint_data['model']
            msg = self.model.load_state_dict(checkpoint_data_model, strict=(not self.settings.finetune_pretrained_model))
            #print(f'msg = {msg}')

            if not self.settings.finetune_pretrained_model:
                print(f'==> Loading optimizer')
                if self.settings.val_only is False:
                    self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer'])
                self.epoch_start = checkpoint_data['epoch'] + 1

                if ('fp16_scaler' in checkpoint_data) and (checkpoint_data['fp16_scaler'] is not None):
                    self.trainer.fp16_scaler.load_state_dict(checkpoint_data['fp16_scaler'])


    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            save_results_path = self.prediction_path if self.settings.save_eval_results else None
            val_result = self.trainer.run(self.epoch_start,
                                          mode='Validation',
                                          print_results=True,
                                          save_results_path=save_results_path)
            # Log metrics when available (skip test split without labels)
            if val_result is not None:
                self._log_metrics(val_result, mode='val', epoch=self.epoch_start)

            cost_time = time.time() - t_start
            if self.recorder is not None:
                self.recorder.logger.info('==== Total cost time: {}'.format(
                    datetime.timedelta(seconds=cost_time)))
            self._log_training_time(cost_time)
            return
        best_val_result = None

        #self.trainer.scheduler.step(self.epoch_start*len(self.trainer.train_loader))

        for epoch in range(self.epoch_start, self.settings.n_epochs):

            # Run one epoch
            train_result = self.trainer.run(epoch, mode='Train')
            self._log_metrics(train_result, mode='train', epoch=epoch)

            # Run validation
            if (epoch % self.settings.val_frequency == 0 or
                epoch == self.settings.n_epochs - 1 or
                epoch == self.epoch_start):
                val_result = self.trainer.run(epoch, mode='Validation')
                self._log_metrics(val_result, mode='val', epoch=epoch)

                # Save the best result (skip if test_split - no metrics available)
                if self.recorder is not None and val_result is not None:
                    self.recorder.logger.info(f'---- Best result after Epoch {epoch+1} ----')
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                'Get better {} model: {}'.format(k, v))
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path, 'best_{}_model.pth'.format(k))
                            saved_path_start = os.path.join(
                                self.recorder.checkpoint_path, 'best_{}_model_from_start_{}.pth'.format(k, self.epoch_start))
                            best_val_result[k] = v
                            self._log_best_metric(k, v)

                            checkpoint_data = {
                                'model': self.model.state_dict(),
                                'optimizer': self.trainer.optimizer.state_dict(),
                                'epoch': epoch,
                                k: v,
                            }

                            if self.trainer.fp16_scaler is not None:
                                checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()

                            torch.save(checkpoint_data, saved_path)
                            if self.epoch_start > 0:
                                torch.save(checkpoint_data, saved_path_start)

            # Save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(self.recorder.checkpoint_path, 'checkpoint.pth')

                checkpoint_data = {
                    'model': self.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'epoch': epoch,
                }
                if self.trainer.fp16_scaler is not None:
                    checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()
                torch.save(checkpoint_data, saved_path)

                # Logging best results
                if best_val_result is not None:
                    log_str = '>>> Best Result: '
                    for k, v in best_val_result.items():
                        log_str += '{}: {} '.format(k, v)
                    log_str += '\n'
                    self.recorder.logger.info(log_str)

        # Print total cost time
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info('=== Total cost time: {}'.format(
                datetime.timedelta(seconds=cost_time)))
        self._log_training_time(cost_time)


def _run_pipeline(
    args: argparse.Namespace,
    settings: Option,
    *,
    forced_run_name: Optional[str] = None,
    run_name_suffix: Optional[str] = None,
    task_name_suffix: Optional[str] = None,
    success_status_detail: Optional[str] = None,
) -> Dict[str, object]:
    """
    Execute a training or evaluation run with MLflow/Discord bookkeeping.
    Returns metadata about the execution.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = getattr(settings, 'mlflow_experiment', None)
    configured_run_name = getattr(settings, 'mlflow_run_name', None)
    mlflow_requested = getattr(settings, 'mlflow_enabled', None)

    if mlflow_requested is None:
        mlflow_requested = mlflow_utils.is_enabled(tracking_uri)

    mlflow_enabled = False
    if mlflow_requested:
        if not tracking_uri:
            raise RuntimeError('MLflow tracking is enabled but MLFLOW_TRACKING_URI environment variable is not set.')
        mlflow_enabled = mlflow_utils.setup(tracking_uri=tracking_uri, experiment=experiment_name)

    base_run_name = mlflow_utils.default_run_name(settings.config.get('model_type', 'rangevit'),
                                                  getattr(settings, 'id', None))
    run_name = forced_run_name or configured_run_name or base_run_name
    if run_name_suffix:
        run_name = f"{run_name}{run_name_suffix}"

    mlflow_context = mlflow_utils.start_run(run_name=run_name) if mlflow_enabled else nullcontext()
    task_name = run_name or getattr(settings, 'id', 'RangeViT')
    if task_name_suffix:
        task_name = f"{task_name} {task_name_suffix}"

    with mlflow_context:
        if mlflow_enabled:
            mlflow_utils.set_tags(mlflow_utils.collect_tags_from_settings(settings))
            mlflow_utils.log_params(mlflow_utils.collect_params_from_settings(settings))
            mlflow_utils.log_input_dataset(getattr(settings, 'dataset', "SemanticKitti"), context="training")

        run_start = time.time()
        start_context_lines = _build_run_context_lines(args, settings)
        context_text = "\n".join(start_context_lines)

        if _is_notification_master():
            post_message(
                f"`{task_name}` started\n{context_text}",
                username="RangeViT Bot",
            )

        try:
            exp = Experiment(settings, mlflow_active=mlflow_enabled)
            exp.run()
        except Exception as exc:
            if _is_notification_master():
                failure_lines = _build_run_context_lines(args, settings) + [f"- Status details: Error: {exc}"]
                notify_run_completion(
                    task_name=task_name,
                    success=False,
                    elapsed_seconds=time.time() - run_start,
                    extra_message="\n".join(failure_lines),
                )
            raise
        else:
            if _is_notification_master():
                detail_text = success_status_detail or f"Outputs saved to {settings.save_path}"
                success_lines = _build_run_context_lines(args, settings) + [f"- Status details: {detail_text}"]
                notify_run_completion(
                    task_name=task_name,
                    success=True,
                    elapsed_seconds=time.time() - run_start,
                    extra_message="\n".join(success_lines),
                )
        finally:
            # Ensure DDP is torn down even on failure
            try:
                tools.cleanup()
            except Exception:
                pass

    return {
        "elapsed_seconds": time.time() - run_start,
        "settings": settings,
        "task_name": task_name,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Options')
    parser.add_argument('config_path', type=str, metavar='config_path',
                        help='path of config file, type: string')
    parser.add_argument('--data_root', type=str, required=True,
                        help='path to the data, type: string')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the file, type: string')
    parser.add_argument('--id', type=str,
                        help='name to identify the run')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of threads used for data loading, type: int')
    parser.add_argument('--pretrained_model', type=str,
                        help='path of pre-trained model to initialize the ViT encoder backbone, type: string')
    parser.add_argument('--checkpoint', type=str,
                        help='path of checkpoint model for resuming training or evaluation, type: string')
    parser.add_argument('--window_stride', type=int,
                        help='sliding window stride during validation, type: int')
    parser.add_argument('--mini', action='store_true', help='use mini version of the dataset, type: bool')
    parser.add_argument('--val_only', action='store_true', help='run inference only')
    parser.add_argument('--test_split', action='store_true', help='run inference on the test split')
    parser.add_argument('--save_eval_results', action='store_true', help='save the predictions')
    parser.add_argument('--full', action='store_true',
                        help='after training, evaluate on the test split using the latest checkpoint and save results')
    parser.add_argument('--log_frequency', type=int, default=100, help='logging frequency')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    settings = _prepare_settings(args)

    _run_pipeline(args, settings, success_status_detail=f"Outputs saved to {settings.save_path}")

    if args.full and not settings.val_only:
        checkpoint_path = os.path.join(settings.save_path, 'checkpoint', 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Full evaluation requested but checkpoint not found at {checkpoint_path}"
            )

        eval_args_dict = vars(args).copy()
        eval_args_dict.update({
            'val_only': True,
            'test_split': True,
            'save_eval_results': True,
            'checkpoint': checkpoint_path,
            'pretrained_model': None,
            'full': False,
        })
        eval_args = argparse.Namespace(**eval_args_dict)
        eval_settings = _prepare_settings(eval_args)

        _run_pipeline(
            eval_args,
            eval_settings,
            run_name_suffix='-test',
            task_name_suffix='[test]',
            success_status_detail=f"Test outputs saved to {eval_settings.save_path}",
        )
