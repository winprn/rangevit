# python3 tools/profile_metrics.py config_kitti_tinyvim.yaml --device cuda --amp --validation_style
# python3 tools/profile_metrics.py config_kitti_trainval.yaml --device cuda --amp --validation_style --profile_kpconv
import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
import types
from types import SimpleNamespace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile RangeTinyViM/RangeViT model parameters, FLOPs, latency, and memory."
    )
    parser.add_argument("config_path", type=str, help="Path to config YAML.")
    parser.add_argument("--data_root", type=str, help="Override data_root from config.")
    parser.add_argument("--save_path", type=str, help="Override save_path from config.")
    parser.add_argument("--id", type=str, help="Optional run id override.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cpu", "cuda"),
        help="Device used for profiling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy batch size for FLOPs and inference profiling.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="train",
        choices=("train", "window", "original"),
        help="Which configured image size to profile.",
    )
    parser.add_argument("--height", type=int, help="Custom input height.")
    parser.add_argument("--width", type=int, help="Custom input width.")
    parser.add_argument(
        "--include_pretrained",
        action="store_true",
        help="Load pretrained weights/checkpoint paths from config.",
    )
    parser.add_argument(
        "--profile_kpconv",
        action="store_true",
        help="For KPConv configs, profile the full 2D+KPConv model.",
    )
    parser.add_argument(
        "--validation_style",
        action="store_true",
        help="Profile validation/inference-style full-scan execution using the repo's inference pipeline.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Dummy point count for full KPConv profiling. Defaults to H*W.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations before latency measurement.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Measured iterations for latency.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use AMP autocast during latency/memory profiling on CUDA.",
    )
    parser.add_argument(
        "--measure_train_vram",
        action="store_true",
        help="Also measure peak VRAM for a surrogate train step (forward + backward).",
    )
    parser.add_argument(
        "--mock_selective_scan",
        action="store_true",
        help="Install a mock selective_scan_cuda fallback if the extension is missing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final metrics as JSON.",
    )
    return parser.parse_args()


def install_mock_selective_scan():
    import torch

    module = types.ModuleType("selective_scan_cuda")

    def mock_fwd(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        saved = torch.empty(0, device=u.device, dtype=u.dtype)
        return u, saved, None

    def mock_bwd(u, delta, A, B, C, D, z, delta_bias, dout, x, out, dz, delta_softplus, recompute_out_z):
        dD = torch.zeros_like(D) if D is not None else None
        ddelta_bias = torch.zeros_like(delta_bias) if delta_bias is not None else None
        return (
            torch.zeros_like(u),
            torch.zeros_like(delta),
            torch.zeros_like(A),
            torch.zeros_like(B),
            torch.zeros_like(C),
            dD,
            ddelta_bias,
        )

    module.fwd = mock_fwd
    module.bwd = mock_bwd
    sys.modules["selective_scan_cuda"] = module


def ensure_selective_scan():
    if importlib.util.find_spec("selective_scan_cuda") is not None:
        return "real"
    install_mock_selective_scan()
    return "mock"


def build_option(args):
    from option import Option

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
    if args.validation_style:
        return int(settings.original_image_size[0]), int(settings.original_image_size[1]), "validation"
    if args.input_mode == "train":
        return int(settings.image_size[0]), int(settings.image_size[1]), "train"
    if args.input_mode == "window":
        return int(settings.window_size[0]), int(settings.window_size[1]), "window"
    if args.input_mode == "original":
        return int(settings.original_image_size[0]), int(settings.original_image_size[1]), "original"
    raise ValueError(f"Unknown input_mode: {args.input_mode}")


class FlopCounter:
    def __init__(self, model):
        import torch.nn as nn
        from models.blocks import Attention as RangeViTAttention

        self.model = model
        self.nn = nn
        self.RangeViTAttention = RangeViTAttention
        self.handles = []
        self.macs = 0
        self.extra_flops = 0

    def _add_macs(self, value):
        self.macs += int(value)

    def _add_flops(self, value):
        self.extra_flops += int(value)

    def _conv_hook(self, module, inputs, output):
        import torch

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
        import torch

        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return
        out_features = output.shape[-1]
        output_elements = output.numel() // out_features
        self._add_macs(output_elements * module.in_features * out_features)

    def _batchnorm_hook(self, module, inputs, output):
        import torch

        if torch.is_tensor(output):
            self._add_flops(2 * output.numel())

    def _layernorm_hook(self, module, inputs, output):
        import torch

        if torch.is_tensor(output):
            self._add_flops(5 * output.numel())

    def _pool_hook(self, module, inputs, output):
        import torch

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
        import torch

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
        nn = self.nn
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
            elif isinstance(module, self.RangeViTAttention):
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
    import torch

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


def make_validation_meta(batch_size):
    return [dict(flip=False) for _ in range(batch_size)]


def forward_validation_style(model, settings, dummy, device, use_amp=False, profile_kpconv=False, num_points=None):
    import torch
    from utils.inference.inference_utils import inference

    if dummy.shape[0] != 1:
        raise ValueError("validation_style profiling currently supports only batch_size=1.")

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else contextlib.nullcontext()
    )
    im_meta = make_validation_meta(dummy.shape[0])
    ori_shape = dummy.shape[2:4]

    with autocast:
        if settings.use_kpconv and profile_kpconv:
            features2d = inference(
                model.rangevit,
                [dummy],
                [im_meta[0]],
                ori_shape=ori_shape,
                window_size=settings.window_size,
                window_stride=settings.window_stride,
                batch_size=dummy.shape[0],
                use_kpconv=True,
                use_sliding_window=settings.use_sliding_window,
            ).unsqueeze(0)
            px, py, pxyz, pknn, num_points_tensor = make_kpconv_inputs(
                batch_size=dummy.shape[0],
                height=dummy.shape[2],
                width=dummy.shape[3],
                num_points=num_points,
                device=device,
            )
            return model.rangevit.kpclassifier(features2d, px, py, pxyz, pknn, num_points_tensor)

        if settings.use_kpconv:
            return inference(
                model.rangevit,
                [dummy],
                [im_meta[0]],
                ori_shape=ori_shape,
                window_size=settings.window_size,
                window_stride=settings.window_stride,
                batch_size=dummy.shape[0],
                use_kpconv=True,
                use_sliding_window=settings.use_sliding_window,
            )

        return inference(
            model.rangevit,
            [dummy],
            [im_meta[0]],
            ori_shape=ori_shape,
            window_size=settings.window_size,
            window_stride=settings.window_stride,
            batch_size=dummy.shape[0],
            use_kpconv=False,
            use_sliding_window=settings.use_sliding_window,
        )


def forward_model(model, settings, dummy, device, use_amp=False, profile_kpconv=False, num_points=None, validation_style=False):
    import torch

    if validation_style:
        return forward_validation_style(
            model=model,
            settings=settings,
            dummy=dummy,
            device=device,
            use_amp=use_amp,
            profile_kpconv=profile_kpconv,
            num_points=num_points,
        )

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast:
        if settings.use_kpconv and profile_kpconv:
            px, py, pxyz, pknn, num_points_tensor = make_kpconv_inputs(
                batch_size=dummy.shape[0],
                height=dummy.shape[2],
                width=dummy.shape[3],
                num_points=num_points,
                device=device,
            )
            return model(dummy, px, py, pxyz, pknn, num_points_tensor)
        if settings.use_kpconv:
            return model.rangevit.forward_2d_features(dummy)
        return model(dummy)


def split_main_output(output):
    import torch

    if isinstance(output, (tuple, list)):
        main = output[0]
        aux = []
        if len(output) > 1:
            aux_item = output[1]
            if isinstance(aux_item, (tuple, list)):
                aux = [x for x in aux_item if torch.is_tensor(x)]
            elif torch.is_tensor(aux_item):
                aux = [aux_item]
        return main, aux
    return output, []


def profile_flops(model, settings, batch_size, height, width, device, use_amp=False, profile_kpconv=False, num_points=None, validation_style=False):
    import torch

    model = model.to(device)
    model.eval()
    profiler = FlopCounter(model)
    profiler.register()
    dummy = torch.randn(batch_size, settings.in_channels, height, width, device=device)
    with torch.no_grad():
        _ = forward_model(
            model,
            settings,
            dummy,
            device,
            use_amp=use_amp,
            profile_kpconv=profile_kpconv,
            num_points=num_points,
            validation_style=validation_style,
        )
    profiler.remove()
    return profiler.macs, profiler.total_flops()


def benchmark_latency(model, settings, batch_size, height, width, device, warmup, iters, use_amp=False, profile_kpconv=False, num_points=None, validation_style=False):
    import torch

    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, settings.in_channels, height, width, device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = forward_model(
                model,
                settings,
                dummy,
                device,
                use_amp=use_amp,
                profile_kpconv=profile_kpconv,
                num_points=num_points,
                validation_style=validation_style,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings_ms = []
        for _ in range(max(1, iters)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = forward_model(
                model,
                settings,
                dummy,
                device,
                use_amp=use_amp,
                profile_kpconv=profile_kpconv,
                num_points=num_points,
                validation_style=validation_style,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000.0)

    mean_ms = sum(timings_ms) / len(timings_ms)
    variance = sum((x - mean_ms) ** 2 for x in timings_ms) / len(timings_ms)
    std_ms = variance ** 0.5
    throughput = (1000.0 / mean_ms) * batch_size if mean_ms > 0 else float("inf")
    return mean_ms, std_ms, throughput


def measure_inference_vram(model, settings, batch_size, height, width, device, use_amp=False, profile_kpconv=False, num_points=None, validation_style=False):
    import torch

    if device.type != "cuda":
        return None

    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, settings.in_channels, height, width, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = forward_model(
            model,
            settings,
            dummy,
            device,
            use_amp=use_amp,
            profile_kpconv=profile_kpconv,
            num_points=num_points,
            validation_style=validation_style,
        )
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def measure_train_vram(model, settings, batch_size, height, width, device, use_amp=False, profile_kpconv=False, num_points=None):
    import torch
    import torch.nn.functional as F

    if device.type != "cuda":
        return None

    model = model.to(device)
    model.train()
    dummy = torch.randn(batch_size, settings.in_channels, height, width, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)

    output = forward_model(
        model,
        settings,
        dummy,
        device,
        use_amp=use_amp,
        profile_kpconv=profile_kpconv,
        num_points=num_points,
    )
    main_out, aux_outs = split_main_output(output)

    if not torch.is_tensor(main_out):
        raise RuntimeError("Training VRAM measurement expects a tensor output.")

    labels = torch.randint(0, settings.n_classes, main_out.shape[:1] + main_out.shape[2:], device=device)
    loss = F.cross_entropy(main_out.float(), labels)
    for aux in aux_outs:
        loss = loss + 0.3 * F.cross_entropy(aux.float(), labels)
    loss.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    model.zero_grad(set_to_none=True)
    return peak


def build_model(settings):
    from main import build_rangevit_model

    return build_rangevit_model(settings, pretrained_path=settings.pretrained_model)


def main():
    args = parse_args()

    selective_scan_mode = "real"
    if importlib.util.find_spec("selective_scan_cuda") is None:
        if not args.mock_selective_scan:
            raise ModuleNotFoundError(
                "selective_scan_cuda is not installed. Re-run with --mock_selective_scan "
                "for approximate profiling, or install the TinyViM selective-scan extension."
            )
        selective_scan_mode = ensure_selective_scan()

    settings = build_option(args)
    height, width, input_label = resolve_input_size(settings, args)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    model = build_model(settings)
    total_params, trainable_params = count_parameters(model)
    macs, flops = profile_flops(
        model=model,
        settings=settings,
        batch_size=args.batch_size,
        height=height,
        width=width,
        device=device,
        use_amp=args.amp,
        profile_kpconv=args.profile_kpconv,
        num_points=args.num_points,
        validation_style=args.validation_style,
    )
    latency_mean_ms, latency_std_ms, throughput = benchmark_latency(
        model=model,
        settings=settings,
        batch_size=args.batch_size,
        height=height,
        width=width,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        use_amp=args.amp,
        profile_kpconv=args.profile_kpconv,
        num_points=args.num_points,
        validation_style=args.validation_style,
    )
    inf_vram = measure_inference_vram(
        model=model,
        settings=settings,
        batch_size=args.batch_size,
        height=height,
        width=width,
        device=device,
        use_amp=args.amp,
        profile_kpconv=args.profile_kpconv,
        num_points=args.num_points,
        validation_style=args.validation_style,
    )
    train_vram = None
    if args.measure_train_vram:
        train_vram = measure_train_vram(
            model=model,
            settings=settings,
            batch_size=args.batch_size,
            height=height,
            width=width,
            device=device,
            use_amp=args.amp,
            profile_kpconv=args.profile_kpconv,
            num_points=args.num_points,
        )

    results = {
        "config": os.path.abspath(args.config_path),
        "device": str(device),
        "input_mode": input_label,
        "input_shape": [args.batch_size, settings.in_channels, height, width],
        "validation_style": bool(args.validation_style),
        "backbone": settings.vit_backbone,
        "decoder": settings.decoder,
        "point_postproc": "kpconv" if settings.use_kpconv else ("knn" if settings.use_knn else "none"),
        "selective_scan_mode": selective_scan_mode,
        "amp": bool(args.amp),
        "parameters_total": int(total_params),
        "parameters_trainable": int(trainable_params),
        "macs": int(macs),
        "flops": int(flops),
        "latency_mean_ms": float(latency_mean_ms),
        "latency_std_ms": float(latency_std_ms),
        "throughput_samples_per_sec": float(throughput),
        "peak_vram_inference_bytes": None if inf_vram is None else int(inf_vram),
        "peak_vram_inference_mb": None if inf_vram is None else float(inf_vram / (1024 ** 2)),
        "peak_vram_train_step_bytes": None if train_vram is None else int(train_vram),
        "peak_vram_train_step_mb": None if train_vram is None else float(train_vram / (1024 ** 2)),
        "notes": [
            "KNN point post-processing is not included in MAC/FLOP/latency/model-VRAM numbers.",
            "validation_style=True measures full-scan inference with config windowing/full-frame behavior.",
            "Hook-based FLOPs undercount custom kernels such as TinyViM selective scan.",
            "If selective_scan_mode=mock, latency and memory are only lower-bound approximations.",
            "Training VRAM uses a surrogate cross-entropy backward pass, not the full project loss stack.",
        ],
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 72)
    print("RangeTinyViM Profile")
    print("=" * 72)
    print(f"Config:                 {results['config']}")
    print(f"Device:                 {results['device']}")
    print(f"Backbone:               {results['backbone']}")
    print(f"Decoder:                {results['decoder']}")
    print(f"Point postproc:         {results['point_postproc']}")
    print(f"Selective scan:         {results['selective_scan_mode']}")
    print(f"Input mode:             {results['input_mode']}")
    print(f"Input tensor:           {results['input_shape']}")
    print(f"Validation style:       {results['validation_style']}")
    print(f"AMP:                    {results['amp']}")
    print(f"Parameters:             {format_large_number(total_params)} ({total_params:,})")
    print(f"Trainable parameters:   {format_large_number(trainable_params)} ({trainable_params:,})")
    print(f"MACs:                   {format_large_number(macs)} ({macs:,})")
    print(f"FLOPs:                  {format_large_number(flops)} ({flops:,})")
    print(f"Latency mean:           {latency_mean_ms:.3f} ms")
    print(f"Latency std:            {latency_std_ms:.3f} ms")
    print(f"Throughput:             {throughput:.3f} samples/s")
    if inf_vram is not None:
        print(f"Peak VRAM (inference):  {inf_vram / (1024 ** 2):.2f} MiB")
    if train_vram is not None:
        print(f"Peak VRAM (train step): {train_vram / (1024 ** 2):.2f} MiB")
    print()
    print("Notes:")
    for note in results["notes"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
