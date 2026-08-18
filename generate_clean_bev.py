import argparse
import os
import sys
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader
from main import build_rangevit_model
from utils.inference.inference_utils import inference
from option import Option

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
    parser = argparse.ArgumentParser(description="Generate clean BEV inference images (only green/red points on white canvas).")
    parser.add_argument("--config", type=str, default="config/kitti/main/config_tinyvim_aug.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/best_miou_model_67_15.pth", help="Path to checkpoint.")
    parser.add_argument("--data_root", type=str, default="../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences", help="Path to SemanticKITTI sequences.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output PNG images.")
    parser.add_argument("--artifact_dir", type=str, help="Also save copies in Gemini chat artifact directory.")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of random frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bev_size", type=int, default=800, help="Size of BEV square canvas.")
    parser.add_argument("--bev_boundary", type=float, default=50.0, help="Max distance in meters for BEV.")
    return parser.parse_args()

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
    
    # Handle paths
    config_path = to_wsl_path(args.config)
    checkpoint_path = to_wsl_path(args.checkpoint)
    data_root = to_wsl_path(args.data_root)
    output_dir = Path(to_wsl_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_dir = None
    if args.artifact_dir:
        artifact_dir = Path(to_wsl_path(args.artifact_dir))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
    # Load configuration
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
    print(f"Selected random indices for clean BEV: {selected_indices}")
    
    H, W = args.bev_size, args.bev_size
    bev_boundary = args.bev_boundary
    
    for i, idx in enumerate(selected_indices):
        seq_id, frame_id = semkitti_ds.parsePathInfoByIndex(idx)
        print(f"Processing [{i+1}/{args.num_frames}] - Frame {frame_id} (index {idx})")
        
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
        
        # Original 3D point cloud coordinates (x, y)
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
        
        # Unproject predictions back to point cloud (fast direct indexing)
        pred_np = pred_argmax[uproj_y_idx, uproj_x_idx].cpu().numpy()
        gt_labels = sem_label.numpy()
        pred_labels = pred_np
        
        # Create white canvas
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        
        # Priority: 0 (unlabeled) -> 1 (correct) -> 2 (incorrect)
        priority = np.zeros_like(gt_labels)
        correct_mask = (pred_labels == gt_labels) & (gt_labels != 0)
        incorrect_mask = (pred_labels != gt_labels) & (gt_labels != 0)
        
        priority[correct_mask] = 1
        priority[incorrect_mask] = 2
        
        # Define colors (BGR for OpenCV)
        colors = np.full((len(gt_labels), 3), 255, dtype=np.uint8)
        colors[gt_labels == 0] = [230, 230, 230] # Light gray for unlabeled
        colors[correct_mask] = [0, 180, 0] # Green for correct
        colors[incorrect_mask] = [0, 0, 255] # Red for incorrect
        
        # Filter points within BEV boundary
        valid = (x >= -bev_boundary) & (x <= bev_boundary) & (y >= -bev_boundary) & (y <= bev_boundary)
        x_v = x[valid]
        y_v = y[valid]
        colors_v = colors[valid]
        priority_v = priority[valid]
        
        # Sort by priority
        sort_idx = np.argsort(priority_v)
        x_v = x_v[sort_idx]
        y_v = y_v[sort_idx]
        colors_v = colors_v[sort_idx]
        
        # Map physical coordinates (x, y) to image grid coordinates (row, col)
        cols = ((y_v + bev_boundary) / (2 * bev_boundary) * (W - 1)).astype(np.int32)
        rows = (((bev_boundary - x_v) / (2 * bev_boundary)) * (H - 1)).astype(np.int32)
        
        # Draw points (3x3 thickness for clear visibility)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = np.clip(rows + dr, 0, H - 1)
                cc = np.clip(cols + dc, 0, W - 1)
                img[rr, cc] = colors_v
                
        # Save image
        output_path = output_dir / f"clean_bev_frame_{frame_id}.png"
        cv2.imwrite(str(output_path), img)
        print(f"Saved clean BEV to {output_path}")
        
        if artifact_dir:
            artifact_path = artifact_dir / f"clean_bev_frame_{frame_id}.png"
            cv2.imwrite(str(artifact_path), img)
            
    print("\nClean BEV visualizations completed successfully!")

if __name__ == "__main__":
    main()
