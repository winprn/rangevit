# Unified Main Script for RangeViT and RangeFormer
# Supports both architectures with automatic model selection

import argparse
from html import parser
import os
import sys
import yaml
import torch
import torch.distributed as dist

from option import Option
from utils.tools import Recorder
import utils.tools as tools
from utils import mlflow_utils


def create_model(settings):
    """
    Create model based on model_type in config.

    Args:
        settings: Option object with configuration

    Returns:
        model: PyTorch model (RangeViT or RangeFormer)
    """
    model_type = settings.config.get('model_type', 'rangevit')
    print(f'\nModel type: {model_type}')

    if model_type == 'rangeformer':
        print('Creating RangeFormer model...')
        from train_rangeformer import create_rangeformer_model
        model = create_rangeformer_model(settings)

    elif model_type == 'rangevit':
        print('Creating RangeViT model...')
        raise NotImplementedError("[main-unified] RangeViT model creation is not implemented in this snippet.")
        from models.rangevit import RangeViT

        # RangeViT model configuration
        model = RangeViT(
            in_channels=settings.in_channels,
            n_cls=settings.n_classes,
            backbone=settings.vit_backbone,
            image_size=settings.image_size,
            pretrained_path=settings.pretrained_model,
            new_patch_size=settings.patch_size,
            new_patch_stride=settings.patch_stride,
            reuse_pos_emb=settings.reuse_pos_emb,
            reuse_patch_emb=settings.reuse_patch_emb,
            conv_stem=settings.conv_stem,
            stem_base_channels=settings.stem_base_channels,
            stem_hidden_dim=settings.D_h,
            skip_filters=settings.skip_filters,
            decoder=settings.decoder,
            use_kpconv=settings.use_kpconv
        )

        # Print model statistics
        stats = model.counter_model_parameters()
        print(f'\nRangeViT Model Configuration:')
        print(f'  Total parameters: {stats["total_num_parameters"]:,}')
        print(f'  Encoder parameters: {stats["encoder_num_parameters"]:,}')
        print(f'  Decoder parameters: {stats["decoder_num_parameters"]:,}')
        print(f'  Stem parameters: {stats["stem_num_parameters"]:,}')

    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model


def create_trainer(settings, model, recorder):
    """
    Create trainer based on model type.

    Args:
        settings: Option object
        model: PyTorch model
        recorder: Recorder object

    Returns:
        trainer: Trainer object
    """
    model_type = settings.config.get('model_type', 'rangevit')

    if model_type == 'rangeformer':
        from train_rangeformer import RangeFormerTrainer
        trainer = RangeFormerTrainer(settings, model, recorder)

    elif model_type == 'rangevit':
        raise NotImplementedError("[main-unified] RangeViT model trainer creation is not implemented in this snippet.")
        from train import Trainer
        trainer = Trainer(settings, model, recorder)

    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return trainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Unified Training for RangeViT and RangeFormer')
    parser.add_argument('config_path', nargs='?', default=None,
                        help='Path to config YAML file (positional alternative to --config)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--data-root', '--data_root', dest='data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--val-only', action='store_true',
                        help='Run validation only')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of distributed process')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='URL for distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='Distributed backend')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the file, type: string')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation or finetuning')
    parser.add_argument('--test_split', action='store_true',
                        help='Run inference on the SemanticKITTI test split (no labels)')
    parser.add_argument('--save_eval_results', action='store_true',
                        help='Save per-scan predictions during evaluation')
    parser.add_argument('--id', type=str, default=None,
                        help='Override run identifier')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    
    args = parser.parse_args()

    # Resolve config path (allow positional or --config)
    resolved_config = args.config if args.config is not None else args.config_path
    if resolved_config is None:
        parser.error('Please provide a config path via --config or as a positional argument.')

    # Initialize settings
    settings = Option(resolved_config, args)
    settings.data_root = args.data_root
    settings.val_only = args.val_only
    settings.distributed = args.distributed
    settings.gpu = args.gpu
    settings.world_size = args.world_size
    settings.rank = args.rank
    settings.dist_url = args.dist_url
    settings.dist_backend = args.dist_backend
    settings.test_split = args.test_split
    settings.save_eval_results = args.save_eval_results
    settings.seed = args.seed
    if args.id is not None:
        settings.id = args.id
        settings.save_path = os.path.join(os.path.dirname(settings.save_path), f'log_{settings.id}')
    if args.checkpoint is not None:
        settings.checkpoint = args.checkpoint
        settings.pretrained_model = None
        settings.finetune_pretrained_model = False
    if settings.val_only:
        settings.save_path = os.path.join(settings.save_path, f'Eval_{settings.id}')

    # Print configuration
    print('\n' + '=' * 60)
    print('Configuration:')
    print('=' * 60)
    print(f'Config file: {resolved_config}')
    print(f'Model type: {settings.config.get("model_type", "rangevit")}')
    print(f'Dataset: {settings.dataset}')
    print(f'Data root: {settings.data_root}')
    print(f'Number of classes: {settings.n_classes}')
    print(f'Image size: {settings.image_size}')
    print(f'Batch size: {settings.batch_size}')
    print(f'Learning rate: {settings.lr}')
    print(f'Epochs: {settings.n_epochs}')
    print(f'Distributed: {settings.distributed}')
    print(f'Mixed precision: {settings.use_fp16}')
    print('=' * 60 + '\n')

    # MLflow setup (optional via env var MLFLOW_TRACKING_URI)
    mlflow_enabled = mlflow_utils.setup() if mlflow_utils.is_enabled() else False
    run_name = mlflow_utils.default_run_name(settings.config.get('model_type', 'rangevit'), getattr(settings, 'id', None))

    # Initialize distributed training
    if settings.distributed:
        if settings.gpu is not None:
            torch.cuda.set_device(settings.gpu)

        dist.init_process_group(
            backend=settings.dist_backend,
            init_method=settings.dist_url,
            world_size=settings.world_size,
            rank=settings.rank
        )

        print(f'Distributed training initialized: rank {settings.rank}/{settings.world_size}')

    # Set seed
    # tools.set_seed(settings.seed)

    # Create recorder (for logging and checkpoints)
    if tools.is_main_process():
        use_tensorboard = not settings.val_only
        recorder = Recorder(settings, save_path=settings.save_path, use_tensorboard=use_tensorboard)
    else:
        recorder = None

    # Create model
    model = create_model(settings)

    # Load checkpoint if specified
    if settings.checkpoint is not None:
        print(f'Loading checkpoint from {settings.checkpoint}')
        checkpoint = torch.load(settings.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Checkpoint loaded from epoch {checkpoint.get("epoch", "unknown")}')
        else:
            model.load_state_dict(checkpoint)
            print('Checkpoint loaded')

    # Create trainer
    trainer = create_trainer(settings, model, recorder)

    # Run training or validation (optionally within MLflow run)
    if mlflow_enabled:
        with mlflow_utils.start_run(run_name=run_name):
            # Log configuration and tags
            mlflow_utils.log_params(mlflow_utils.collect_params_from_settings(settings))
            mlflow_utils.set_tags(mlflow_utils.collect_tags_from_settings(settings))
            # Log the config file as artifact for reproducibility
            try:
                mlflow_utils.log_artifact(resolved_config, artifact_path="config")
            except Exception:
                pass

            if settings.val_only:  # Validation only
                print('\nRunning validation only...')
                val_metrics = trainer.validate(epoch=0)
                if val_metrics is not None and 'miou' in val_metrics:
                    print(f'\nValidation Results:')
                    print(f'  mIoU: {val_metrics["miou"]:.4f}')
                    print(f'  Accuracy: {val_metrics["acc"]:.4f}')
                elif settings.test_split:
                    if settings.save_eval_results and getattr(trainer, 'prediction_path', None):
                        print(f'\nTest predictions saved to: {trainer.prediction_path}')
                    else:
                        print('\nTest split inference completed (predictions not saved).')
            else:  # Full training
                print('\nStarting training...')
                trainer.train()
    else:
        # Fallback without MLflow
        if settings.val_only:  # Validation only
            print('\nRunning validation only...')
            val_metrics = trainer.validate(epoch=0)
            if val_metrics is not None and 'miou' in val_metrics:
                print(f'\nValidation Results:')
                print(f'  mIoU: {val_metrics["miou"]:.4f}')
                print(f'  Accuracy: {val_metrics["acc"]:.4f}')
            elif settings.test_split:
                if settings.save_eval_results and getattr(trainer, 'prediction_path', None):
                    print(f'\nTest predictions saved to: {trainer.prediction_path}')
                else:
                    print('\nTest split inference completed (predictions not saved).')
        else:  # Full training
            print('\nStarting training...')
            trainer.train()

    # Cleanup
    if settings.distributed:
        dist.destroy_process_group()

    print('\nDone!')


if __name__ == '__main__':
    main()
