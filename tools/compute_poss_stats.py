"""Compute per-channel mean/std and raw-label frequencies for SemanticPOSS.

Usage:
    python tools/compute_poss_stats.py \
        --data_root /path/to/SemanticPOSS/sequences \
        --sequences 0 1 2 4 5

Prints YAML-ready img_mean, img_stds, and `content:` entries for
config_poss.yaml and dataset/semantic_poss/semantic-poss.yaml.
"""
import argparse
import os

import numpy as np


H, W = 40, 1800
N_CHANNELS = 5  # range, x, y, z, intensity


def iter_frames(data_root, sequences):
    for seq in sequences:
        seq_str = f"{int(seq):02d}"
        seq_dir = os.path.join(data_root, seq_str)
        pc_dir = os.path.join(seq_dir, "velodyne")
        lbl_dir = os.path.join(seq_dir, "labels")
        tag_dir = os.path.join(seq_dir, "tag")
        pc_files = sorted(f for f in os.listdir(pc_dir) if f.endswith(".bin"))
        for f in pc_files:
            stem = f.rsplit(".", 1)[0]
            yield (
                os.path.join(pc_dir, f),
                os.path.join(lbl_dir, stem + ".label"),
                os.path.join(tag_dir, stem + ".tag"),
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sequences", type=int, nargs="+", required=True)
    args = ap.parse_args()

    total = 0
    sum_ = np.zeros(N_CHANNELS, dtype=np.float64)
    sumsq = np.zeros(N_CHANNELS, dtype=np.float64)
    label_counts = {}

    for pc_path, lbl_path, tag_path in iter_frames(args.data_root, args.sequences):
        pts = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        tag = np.fromfile(tag_path, dtype=np.uint8).astype(bool)
        if tag.sum() != pts.shape[0]:
            print(f"WARN: tag mismatch for {pc_path}; skipping")
            continue
        rng = np.linalg.norm(pts[:, :3], axis=1).astype(np.float32)
        feats = np.stack([rng, pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]], axis=1).astype(np.float64)
        total += feats.shape[0]
        sum_ += feats.sum(axis=0)
        sumsq += (feats ** 2).sum(axis=0)

        raw = np.fromfile(lbl_path, dtype=np.uint32) & 0xFFFF
        ids, counts = np.unique(raw, return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            label_counts[int(i)] = label_counts.get(int(i), 0) + int(c)

    mean = sum_ / max(total, 1)
    var = sumsq / max(total, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    print("# Paste into config_poss.yaml under sensor:")
    print(f"img_mean: {[float(round(m, 4)) for m in mean.tolist()]}")
    print(f"img_stds: {[float(round(s, 4)) for s in std.tolist()]}")

    total_labels = sum(label_counts.values())
    print("\n# Paste into dataset/semantic_poss/semantic-poss.yaml under content:")
    print("content:")
    for raw_id in sorted(label_counts.keys()):
        freq = label_counts[raw_id] / max(total_labels, 1)
        print(f"  {raw_id}: {freq:.6f}")


if __name__ == "__main__":
    main()
