import argparse
import contextlib
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import build_rangevit_model
from models.blocks import Attention as RangeViTAttention
from option import Option


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile RangeViT model MACs/FLOPs from a config file."
    )
    parser.add_argument("config_path", type=str, help="Path to config YAML.")
    parser.add_argument(
        "--data_root",
        type=str,
        help="Override data_root from config. Only needed if config omits it.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Override save_path from config. Only needed if config omits it.",
    )
    parser.add_argument("--id", type=str, help="Optional run id override.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for profiling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy batch size used for profiling.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="train",
        choices=("train", "window", "original"),
        help="Which configured image size to profile.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Custom input height. Overrides input_mode when provided together with --width.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Custom input width. Overrides input_mode when provided together with --height.",
    )
    parser.add_argument(
        "--include_pretrained",
        action="store_true",
        help="Load pretrained weights/checkpoint paths from config. Not needed for FLOPs.",
    )
    parser.add_argument(
        "--profile_kpconv",
        action="store_true",
        help="For KPConv configs, profile the full 2D+KPConv model instead of only the 2D trunk.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Dummy point count for full KPConv profiling. Defaults to H*W.",
    )
    return parser.parse_args()


def build_option(args):
    option_args = SimpleNamespace(
        save_path=args.save_path,
        data_root=args.data_root,
        id=args.id,
        num_workers=0,
        pretrained_model=None,
        checkpoint=None,
        window_stride=None,
        mini=False,
        val_only=False,
        test_split=False,
        save_eval_results=False,
        log_frequency=100,
        seed=1,
        full=False,
    )
    settings = Option(args.config_path, option_args)
    settings.id = args.id if args.id is not None else settings.id
    settings.data_root = args.data_root if args.data_root is not None else settings.data_root
    settings.save_path = settings.save_path
    settings.num_workers = 0
    settings.val_only = False
    settings.test_split = False
    settings.save_eval_results = False
    settings.log_frequency = 100
    settings.seed = 1

    if not args.include_pretrained:
        settings.pretrained_model = None
        settings.checkpoint = None
        settings.finetune_pretrained_model = False

    return settings


def resolve_input_size(settings, args):
    if (args.height is None) ^ (args.width is None):
        raise ValueError("Provide both --height and --width together.")
    if args.height is not None and args.width is not None:
        return int(args.height), int(args.width), "custom"

    if args.input_mode == "train":
        return int(settings.image_size[0]), int(settings.image_size[1]), "train"
    if args.input_mode == "window":
        return int(settings.window_size[0]), int(settings.window_size[1]), "window"
    if args.input_mode == "original":
        return int(settings.original_image_size[0]), int(settings.original_image_size[1]), "original"
    raise ValueError(f"Unknown input_mode: {args.input_mode}")


class FlopCounter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.macs = 0
        self.extra_flops = 0

    def _add_macs(self, value):
        self.macs += int(value)

    def _add_flops(self, value):
        self.extra_flops += int(value)

    def _conv_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_spatial = 1
        for dim in output.shape[2:]:
            out_spatial *= dim

        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= k
        in_channels = module.in_channels // module.groups
        macs_per_out = kernel_ops * in_channels
        self._add_macs(batch * out_channels * out_spatial * macs_per_out)

    def _conv_transpose_hook(self, module, inputs, output):
        self._conv_hook(module, inputs, output)

    def _linear_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return
        out_features = output.shape[-1]
        output_elements = output.numel() // out_features
        self._add_macs(output_elements * module.in_features * out_features)

    def _batchnorm_hook(self, module, inputs, output):
        if torch.is_tensor(output):
            self._add_flops(2 * output.numel())

    def _layernorm_hook(self, module, inputs, output):
        if torch.is_tensor(output):
            # Mean/var/affine approximation.
            self._add_flops(5 * output.numel())

    def _pool_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return
        kernel_size = module.kernel_size
        if isinstance(kernel_size, int):
            kernel_elems = kernel_size * kernel_size
        else:
            kernel_elems = 1
            for k in kernel_size:
                kernel_elems *= k
        self._add_flops(output.numel() * kernel_elems)

    def _attention_hook(self, module, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        batch, tokens, channels = x.shape
        heads = module.heads
        head_dim = channels // heads
        attn_macs = 2 * batch * heads * tokens * tokens * head_dim
        self._add_macs(attn_macs)
        softmax_flops = 3 * batch * heads * tokens * tokens
        self._add_flops(softmax_flops)

    def register(self):
        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                self.handles.append(module.register_forward_hook(self._conv_transpose_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.handles.append(module.register_forward_hook(self._batchnorm_hook))
            elif isinstance(module, nn.LayerNorm):
                self.handles.append(module.register_forward_hook(self._layernorm_hook))
            elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                self.handles.append(module.register_forward_hook(self._pool_hook))
            elif isinstance(module, RangeViTAttention):
                self.handles.append(module.register_forward_hook(self._attention_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def total_flops(self):
        return (2 * self.macs) + self.extra_flops


def format_large_number(value):
    if value >= 1e12:
        return f"{value / 1e12:.4f} T"
    if value >= 1e9:
        return f"{value / 1e9:.4f} G"
    if value >= 1e6:
        return f"{value / 1e6:.4f} M"
    return str(value)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_kpconv_inputs(batch_size, height, width, num_points, device):
    if batch_size != 1:
        raise ValueError("Full KPConv profiling currently supports only batch_size=1.")
    if num_points is None:
        num_points = height * width
    px = torch.rand(num_points, device=device) * 2.0 - 1.0
    py = torch.rand(num_points, device=device) * 2.0 - 1.0
    pxyz = torch.randn(num_points, 3, device=device)
    knn_k = 7
    pknn = torch.randint(0, num_points, (num_points, knn_k), device=device)
    num_points_tensor = torch.tensor([num_points], device=device, dtype=torch.long)
    return px, py, pxyz, pknn, num_points_tensor


def profile_model(model, settings, batch_size, height, width, device, profile_kpconv=False, num_points=None):
    model = model.to(device)
    model.eval()
    profiler = FlopCounter(model)
    profiler.register()
    dummy = torch.randn(batch_size, settings.in_channels, height, width, device=device)

    with torch.no_grad():
        with contextlib.nullcontext():
            if settings.use_kpconv and profile_kpconv:
                px, py, pxyz, pknn, num_points_tensor = make_kpconv_inputs(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    num_points=num_points,
                    device=device,
                )
                _ = model(dummy, px, py, pxyz, pknn, num_points_tensor)
            elif settings.use_kpconv:
                _ = model.rangevit.forward_2d_features(dummy)
            else:
                _ = model(dummy)

    profiler.remove()
    return profiler.macs, profiler.total_flops()


def main():
    args = parse_args()
    settings = build_option(args)
    height, width, input_label = resolve_input_size(settings, args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    model = build_rangevit_model(settings, pretrained_path=settings.pretrained_model)
    total_params, trainable_params = count_parameters(model)
    macs, flops = profile_model(
        model=model,
        settings=settings,
        batch_size=args.batch_size,
        height=height,
        width=width,
        device=device,
        profile_kpconv=args.profile_kpconv,
        num_points=args.num_points,
    )

    print("=" * 72)
    print("RangeViT Config Profile")
    print("=" * 72)
    print(f"Config:           {os.path.abspath(args.config_path)}")
    print(f"Backbone:         {settings.vit_backbone}")
    print(f"Decoder:          {settings.decoder}")
    print(f"Point postproc:   {'kpconv' if settings.use_kpconv else ('knn' if settings.use_knn else 'none')}")
    print(f"Input mode:       {input_label}")
    print(f"Input tensor:     [{args.batch_size}, {settings.in_channels}, {height}, {width}]")
    if settings.use_kpconv:
        mode_name = "full model" if args.profile_kpconv else "2D trunk only"
        print(f"KPConv profile:   {mode_name}")
        if args.profile_kpconv:
            effective_points = args.num_points if args.num_points is not None else height * width
            print(f"Dummy points:     {effective_points}")
    print(f"Parameters:       {format_large_number(total_params)} ({total_params:,})")
    print(f"Trainable params: {format_large_number(trainable_params)} ({trainable_params:,})")
    print(f"MACs:             {format_large_number(macs)} MACs ({macs / 1e9:.4f} GMACs)")
    print(f"FLOPs:            {format_large_number(flops)} FLOPs ({flops / 1e9:.4f} GFLOPs)")
    print()
    print("Notes:")
    print("- FLOPs are reported as 2 x MACs plus lightweight ops counted from norms/pooling/softmax.")
    print("- TinyViM custom selective-scan CUDA ops are not explicitly modeled and may be undercounted.")
    print("- Pretrained weights are ignored by default because they do not change the operation count.")


if __name__ == "__main__":
    main()
