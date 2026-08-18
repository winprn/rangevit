import argparse
import os
import sys
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
import utils.postproc
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
    parser = argparse.ArgumentParser(description="Compile BEV validation video for RangeViM predictions on SemanticKITTI sequence 08.")
    parser.add_argument("--config", type=str, default="config/kitti/main/config_tinyvim_aug.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/best_miou_model_67_15.pth", help="Path to checkpoint.")
    parser.add_argument("--data_root", type=str, default="../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences", help="Path to SemanticKITTI sequences.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output MP4 video.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of frames to compile (default -1 for all).")
    parser.add_argument("--knn", action="store_true", default=False, help="Use KNN post-processing (slower).")
    parser.add_argument("--knn_search", type=int, default=7, help="KNN search window.")
    parser.add_argument("--bev_size", type=int, default=800, help="Size of BEV square canvas.")
    parser.add_argument("--bev_boundary", type=float, default=50.0, help="Max distance in meters for BEV.")
    return parser.parse_args()

def build_learning_map_inv_lut(learning_map_inv: dict) -> np.ndarray:
    max_key = max(int(k) for k in learning_map_inv.keys())
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map_inv.items():
        lut[int(k)] = int(v)
    return lut

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
    
    # Handle paths
    config_path = to_wsl_path(args.config)
    checkpoint_path = to_wsl_path(args.checkpoint)
    data_root = to_wsl_path(args.data_root)
    output_path = to_wsl_path(args.output)
    
    # Ensure parent dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
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
    if args.limit > 0:
        num_samples = min(num_samples, args.limit)
    print(f"Compiling video for sequence 08 with {num_samples} frames.")
    
    # Setup Video Writer
    # We use 'mp4v' for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = args.bev_size, args.bev_size
    video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (W, H))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open VideoWriter for path {output_path}")
        sys.exit(1)
        
    # Set up KNN postprocessor if needed
    knn_post = None
    if args.knn:
        knn_params = {
            'knn': 5,
            'search': args.knn_search,
            'sigma': 1.0,
            'cutoff': 1.0,
        }
        knn_post = utils.postproc.KNN(params=knn_params, nclasses=settings.n_classes)
        print("Using KNN post-processing.")
    else:
        print("Using direct projection (no KNN) for fast compilation.")
        
    bev_boundary = args.bev_boundary
    
    # Warmup and starting main loop
    print("Starting video compilation loop...")
    for idx in range(num_samples):
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
        
        # Unproject predictions back to point cloud
        if args.knn and device.type == "cuda" and knn_post is not None:
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
            # Direct indexing
            pred_np = pred_argmax[uproj_y_idx, uproj_x_idx].cpu().numpy()
            
        gt_labels = sem_label.numpy()
        pred_labels = pred_np
        
        # Create white background canvas (BGR format)
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        
        # Identify correct vs incorrect points (exclude unlabeled = 0)
        # Priority: 0 (unlabeled) -> 1 (correct) -> 2 (incorrect)
        priority = np.zeros_like(gt_labels)
        correct_mask = (pred_labels == gt_labels) & (gt_labels != 0)
        incorrect_mask = (pred_labels != gt_labels) & (gt_labels != 0)
        
        priority[correct_mask] = 1
        priority[incorrect_mask] = 2
        
        # Define colors (BGR for OpenCV)
        colors = np.full((len(gt_labels), 3), 255, dtype=np.uint8) # Default white (won't display if filtered)
        colors[gt_labels == 0] = [220, 220, 220] # Light gray for unlabeled
        colors[correct_mask] = [0, 180, 0] # Green for correct (BGR: Blue=0, Green=180, Red=0)
        colors[incorrect_mask] = [0, 0, 255] # Red for incorrect (BGR: Blue=0, Green=0, Red=255)
        
        # Filter points within BEV boundary
        valid = (x >= -bev_boundary) & (x <= bev_boundary) & (y >= -bev_boundary) & (y <= bev_boundary)
        x_v = x[valid]
        y_v = y[valid]
        colors_v = colors[valid]
        priority_v = priority[valid]
        
        # Sort by priority so incorrect points are drawn last (on top)
        sort_idx = np.argsort(priority_v)
        x_v = x_v[sort_idx]
        y_v = y_v[sort_idx]
        colors_v = colors_v[sort_idx]
        
        # Map physical coordinates (x, y) to image grid coordinates (row, col)
        cols = ((y_v + bev_boundary) / (2 * bev_boundary) * (W - 1)).astype(np.int32)
        rows = (((bev_boundary - x_v) / (2 * bev_boundary)) * (H - 1)).astype(np.int32)
        
        # Draw points with 3x3 thickness to make them visible in video
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = np.clip(rows + dr, 0, H - 1)
                cc = np.clip(cols + dc, 0, W - 1)
                img[rr, cc] = colors_v
                
        # Draw text overlay: Frame and legend
        # Legend: Green=Correct, Red=Incorrect, Gray=Unlabeled
        cv2.putText(img, f"Seq 08 - Frame: {idx:06d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "Legend:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "Correct (Green)", (110, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        cv2.putText(img, "Incorrect (Red)", (280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "Unlabeled (Gray)", (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        
        # Write to video
        video_writer.write(img)
        
        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
            print(f"Processed {idx + 1}/{num_samples} frames...")
            
    # Clean up
    video_writer.release()
    print(f"Successfully compiled video and saved to {output_path}!")

if __name__ == "__main__":
    main()
