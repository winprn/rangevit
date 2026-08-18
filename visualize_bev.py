import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader
from main import build_rangevit_model
from utils.inference.inference_utils import inference
import utils.postproc
from option import Option

# Explicit (raw) SemanticKITTI label -> RGB mapping (0-255)
SEMANTIC_KITTI_COLORS = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
}

DATA_CONFIG_PATH = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"

def to_wsl_path(path_str):
    if not path_str:
        return path_str
    # Convert Windows path like C:\Users... to WSL path /mnt/c/Users...
    if ':' in path_str:
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        if rest.startswith('/'):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return path_str

def parse_args():
    parser = argparse.ArgumentParser(description="Generate BEV (Bird's Eye View) inference images for RangeViM.")
    parser.add_argument("--config", type=str, default="config/kitti/main/config_tinyvim_aug.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/best_miou_model_67_15.pth", help="Path to checkpoint.")
    parser.add_argument("--data_root", type=str, default="../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences", help="Path to SemanticKITTI sequences.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save BEV images.")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of random frames to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--knn", action="store_true", default=True, help="Use KNN post-processing.")
    parser.add_argument("--knn_search", type=int, default=7, help="KNN search window.")
    return parser.parse_args()

def build_learning_map_inv_lut(learning_map_inv: dict) -> np.ndarray:
    max_key = max(int(k) for k in learning_map_inv.keys())
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map_inv.items():
        lut[int(k)] = int(v)
    return lut

def build_color_lut(color_map: dict, min_size: int = 0) -> np.ndarray:
    max_key = max(max(int(k) for k in color_map.keys()), min_size)
    lut = np.zeros((max_key + 1, 3), dtype=np.uint8)
    for k, rgb in color_map.items():
        lut[int(k)] = np.array(rgb, dtype=np.uint8)
    return lut

def colorize_labels(mapped_labels: np.ndarray, learning_map_inv_lut: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
    raw_labels = learning_map_inv_lut[mapped_labels]
    raw_labels = np.clip(raw_labels, 0, color_lut.shape[0] - 1)
    colors = color_lut[raw_labels]
    return colors

def load_model_for_inference(settings, device: torch.device):
    model = build_rangevit_model(settings, pretrained_path=settings.pretrained_model)
    checkpoint_path = settings.checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint load status: {msg}")
    return model.to(device)

def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Handle paths (convert Windows to WSL paths if running in WSL environment)
    config_path = to_wsl_path(args.config)
    checkpoint_path = to_wsl_path(args.checkpoint)
    data_root = to_wsl_path(args.data_root)
    output_dir = Path(to_wsl_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print(f"Loading config from {config_path}")
    config = yaml.safe_load(open(config_path, "r"))
    
    # Set up dummy args object for Option parsing
    class DummyArgs:
        pass
    dummy_args = DummyArgs()
    dummy_args.save_path = "./save_val_bev"
    
    settings = Option(config_path, dummy_args)
    settings.checkpoint = checkpoint_path
    settings.data_root = data_root
    
    # Force GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    # Load Model
    model = load_model_for_inference(settings, device)
    model.eval()
    
    # Load dataset
    dataset_cfg = yaml.safe_load(open(to_wsl_path(str(DATA_CONFIG_PATH)), "r"))
    learning_map_inv_lut = build_learning_map_inv_lut(dataset_cfg["learning_map_inv"])
    color_lut = build_color_lut(SEMANTIC_KITTI_COLORS, min_size=int(learning_map_inv_lut.max()))
    
    print(f"Loading SemanticKITTI sequences from {data_root}")
    semkitti_ds = SemanticKitti(
        root=str(data_root),
        sequences=[8], # validation sequence 08
        config_path=to_wsl_path(str(DATA_CONFIG_PATH)),
        has_label=True,
    )
    
    loader = RangeViewLoader(
        dataset=semkitti_ds,
        config=config,
        is_train=False,
        return_uproj=True,
        use_kpconv=False,
    )
    
    num_samples = len(loader)
    print(f"Dataset sequence 08 contains {num_samples} frames.")
    
    # Randomly select frames
    selected_indices = sorted(random.sample(range(num_samples), args.num_frames))
    print(f"Selected random indices for visualization: {selected_indices}")
    
    # Set up KNN postprocessor
    knn_params = {
        'knn': 5,
        'search': args.knn_search,
        'sigma': 1.0,
        'cutoff': 1.0,
    }
    knn_post = utils.postproc.KNN(params=knn_params, nclasses=settings.n_classes)
    
    plt.style.use('dark_background')
    
    for i, idx in enumerate(selected_indices):
        seq_id, frame_id = semkitti_ds.parsePathInfoByIndex(idx)
        print(f"\nProcessing [{i+1}/{args.num_frames}] - Frame {frame_id} (index {idx})")
        
        # Load frame data
        (
            proj_feature_tensor,
            proj_sem_label_tensor,
            proj_mask_tensor,
            proj_range,
            uproj_x_idx,
            uproj_y_idx,
            uproj_depth,
            sem_label,
        ) = loader[idx]
        
        # Original 3D point cloud
        pointcloud, _, _ = semkitti_ds.loadDataByIndex(idx)
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        
        # Model forward pass
        input_feature = proj_feature_tensor.unsqueeze(0).to(device)
        im_meta = {"flip": False}
        
        with torch.no_grad():
            seg_map = inference(
                model.rangevit,
                [input_feature],
                [im_meta],
                ori_shape=input_feature.shape[2:4],
                window_size=settings.window_size,
                window_stride=settings.window_stride,
                batch_size=1,
                use_kpconv=settings.use_kpconv,
                use_sliding_window=settings.use_sliding_window,
            )
            
        pred_output = seg_map.unsqueeze(0) # 1 x n_cls x H x W
        pred_output = F.softmax(pred_output, dim=1)
        pred_argmax = pred_output[0].argmax(dim=0)
        
        # Unproject predictions back to point cloud
        if args.knn and device.type == "cuda":
            # Move data to GPU for fast KNN computation
            proj_depth_gpu = proj_range.to(device)
            uproj_depth_gpu = uproj_depth.to(device)
            pred_argmax_gpu = pred_argmax.to(device)
            uproj_x_idx_gpu = uproj_x_idx.to(device)
            uproj_y_idx_gpu = uproj_y_idx.to(device)
            
            unproj_argmax = knn_post(
                proj_depth_gpu,
                uproj_depth_gpu,
                pred_argmax_gpu,
                uproj_x_idx_gpu,
                uproj_y_idx_gpu,
            )
            pred_np = unproj_argmax.cpu().numpy()
        else:
            # CPU fallback (simple indexing)
            pred_np = pred_argmax[uproj_y_idx, uproj_x_idx].numpy()
            
        gt_labels = sem_label.numpy()
        pred_labels = pred_np
        
        # Colorize labels
        gt_colors = colorize_labels(gt_labels, learning_map_inv_lut, color_lut)
        pred_colors = colorize_labels(pred_labels, learning_map_inv_lut, color_lut)
        
        # Filter points within BEV boundary (-50m to 50m)
        bev_boundary = 50.0
        mask = (x >= -bev_boundary) & (x <= bev_boundary) & (y >= -bev_boundary) & (y <= bev_boundary)
        
        x_filtered = x[mask]
        y_filtered = y[mask]
        gt_colors_filtered = gt_colors[mask]
        pred_colors_filtered = pred_colors[mask]
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
        
        # Ground Truth BEV
        axes[0].scatter(y_filtered, x_filtered, c=gt_colors_filtered/255.0, s=0.1, marker='.')
        axes[0].set_title(f"Ground Truth BEV - Frame {frame_id}", color='white', fontsize=16, pad=15)
        axes[0].set_xlim(-bev_boundary, bev_boundary)
        axes[0].set_ylim(-bev_boundary, bev_boundary)
        axes[0].set_aspect('equal')
        axes[0].set_facecolor('black')
        axes[0].axis('off')
        
        # Prediction BEV
        axes[1].scatter(y_filtered, x_filtered, c=pred_colors_filtered/255.0, s=0.1, marker='.')
        axes[1].set_title(f"RangeViM Prediction BEV - Frame {frame_id}", color='white', fontsize=16, pad=15)
        axes[1].set_xlim(-bev_boundary, bev_boundary)
        axes[1].set_ylim(-bev_boundary, bev_boundary)
        axes[1].set_aspect('equal')
        axes[1].set_facecolor('black')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        output_path = output_dir / f"bev_inference_frame_{frame_id}.png"
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved BEV plot to {output_path}")

    print("\nVisualizations completed successfully!")

if __name__ == "__main__":
    main()
