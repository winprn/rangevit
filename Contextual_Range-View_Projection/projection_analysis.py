import os
import numpy as np
import yaml
from auxiliary.laserscan import SemLaserScan

# =========================
# CONFIG (EDIT THIS)
# =========================
dataset_path = "kitti_dataset/dataset"   # <-- CHANGE THIS
sequence = "00"
config_path = "kitti_dataset/dataset/semantic-kitti.yaml"

max_frames = 4000  # limit for speed

# =========================
# LOAD CONFIG
# =========================
with open(config_path, 'r') as f:
    CFG = yaml.safe_load(f)

color_dict = CFG["color_map"]
num_classes = max(color_dict.keys()) + 1

# =========================
# PATHS
# =========================
scan_dir = os.path.join(dataset_path, "sequences", sequence, "velodyne")
label_dir = os.path.join(dataset_path, "sequences", sequence, "labels")

scan_files = sorted([
    os.path.join(scan_dir, f)
    for f in os.listdir(scan_dir)
    if f.endswith(".bin")
])

label_files = sorted([
    os.path.join(label_dir, f)
    for f in os.listdir(label_dir)
    if f.endswith(".label")
])

# =========================
# METRICS
# =========================
class_depth_count = np.zeros(num_classes)
class_cap_count = np.zeros(num_classes)

# =========================
# MAIN LOOP
# =========================
for scan_file, label_file in zip(scan_files[:max_frames], label_files[:max_frames]):

    base = os.path.basename(scan_file).replace(".bin", "")
    center_file = os.path.join(label_dir, base + ".center.npy")
    weight_file=os.path.join(label_dir, base + ".weight.npy")

    # ---- DEPTH PROJECTION ----
    scan_depth = SemLaserScan(
        color_dict, project=True, scan_proj=True, use_center=False
    )
    scan_depth.open_scan(scan_file)
    scan_depth.open_label(label_file)

    # ---- CAP PROJECTION ----
    scan_cap = SemLaserScan(
        color_dict, project=True, scan_proj=True, use_center=True, use_weight=False
    )
    scan_cap.open_scan(scan_file, center_file=center_file)
    # scan_cap.open_scan(scan_file, weight_file=weight_file)
    scan_cap.open_label(label_file)

    # ---- PIXEL LABELS (H x W) ----
    labels_depth = scan_depth.proj_sem_label
    labels_cap = scan_cap.proj_sem_label

    # valid pixels only (ignore empty pixels)
    valid_mask = labels_depth >= 0

    # =========================
    # COUNT PER CLASS
    # =========================
    for c in range(num_classes):
        class_depth_count[c] += np.sum((labels_depth == c) & valid_mask)
        class_cap_count[c] += np.sum((labels_cap == c) & valid_mask)

# =========================
# RESULTS
# =========================
print("\n===== PER-CLASS RETENTION ANALYSIS =====")

for c in range(num_classes):
    x = class_depth_count[c]   # depth
    y = class_cap_count[c]     # CAP

    if x > 0:
        ratio = y / x
        diff = y - x

        print(f"Class {c}:")
        print(f"  Depth pixels: {int(x)}")
        print(f"  CAP pixels:   {int(y)}")
        print(f"  Ratio (CAP/Depth): {ratio:.3f}")
        print(f"  Difference: {int(diff)}\n")

# =========================
# OPTIONAL: GLOBAL SUMMARY
# =========================
total_depth = np.sum(class_depth_count)
total_cap = np.sum(class_cap_count)

print("===== GLOBAL SUMMARY =====")
print(f"Total pixels (Depth): {int(total_depth)}")
print(f"Total pixels (CAP):   {int(total_cap)}")
print(f"Overall ratio: {total_cap / (total_depth + 1e-6):.4f}")