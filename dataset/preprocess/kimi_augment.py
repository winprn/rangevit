import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


class KimiAugmentor:
    """
    Implements Kimi augmentation: copy real point clusters of minor classes into
    other frames to increase their effective frequency.
    The augmenter scans the dataset once (with caching) to discover rare classes
    and the frames that contain them, then samples and inserts clusters at
    training time with a user-defined probability.
    """

    def __init__(
        self,
        dataset,
        label_mapper,
        n_classes: int,
        kimi_config: Dict,
        dataset_name: str = "dataset",
    ):
        self.dataset = dataset
        self.label_mapper = label_mapper
        self.n_classes = n_classes
        self.prob = float(kimi_config.get("prob", 0.0))
        self.min_points = int(kimi_config.get("min_points", 15))
        self.max_clusters_per_frame = int(kimi_config.get("max_clusters_per_frame", 1))
        translation_std = kimi_config.get("translation_std", [1.0, 1.0, 0.3])
        if len(translation_std) != 3:
            translation_std = [1.0, 1.0, 0.3]
        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.rotation_deg = float(kimi_config.get("rotation_deg", 10.0))
        scale_range = kimi_config.get("scale_range", [0.95, 1.05])
        if not (isinstance(scale_range, (list, tuple)) and len(scale_range) == 2):
            scale_range = [1.0, 1.0]
        self.scale_range = scale_range
        self.anchor_percentile = float(kimi_config.get("anchor_percentile", 5.0))
        self.ignore_classes: Sequence[int] = kimi_config.get("ignore_classes", [0])
        self.minor_freq_ratio = float(kimi_config.get("minor_freq_ratio", 0.01))
        self.max_minor_classes = kimi_config.get("max_minor_classes", None)
        self.max_donor_tries = int(kimi_config.get("max_donor_tries", 5))
        cache_dir = kimi_config.get("cache_dir", "./cache")
        os.makedirs(cache_dir, exist_ok=True)
        dataset_tag = kimi_config.get("dataset_tag", dataset_name)
        cache_name = f"kimi_stats_{dataset_tag}_{len(self.dataset)}.json"
        self.cache_path = os.path.join(cache_dir, cache_name)

        self.cls_freq: np.ndarray
        self.class_to_frames: Dict[int, List[int]]
        self.cls_freq, self.class_to_frames = self._load_or_compute_stats()
        self.minor_classes = self._select_minor_classes(self.cls_freq)
        if len(self.minor_classes) == 0:
            self.prob = 0.0  # disable augmentation if no minor classes found

    def maybe_apply(self, pointcloud: np.ndarray, sem_label: np.ndarray, target_idx: int):
        if self.prob <= 0.0:
            return pointcloud, sem_label
        augmented_points = pointcloud
        augmented_labels = sem_label
        for _ in range(self.max_clusters_per_frame):
            if random.random() > self.prob:
                break
            cls_id = self._sample_minor_class()
            if cls_id is None:
                continue
            donor_idx = self._sample_donor_frame(cls_id, target_idx)
            if donor_idx is None:
                continue
            donor_points, donor_labels_raw, _ = self.dataset.loadDataByIndex(donor_idx)
            donor_labels_mapped = self.label_mapper(donor_labels_raw)
            mask = donor_labels_mapped == cls_id
            if mask.sum() < self.min_points:
                continue
            cluster_points = donor_points[mask].copy()
            cluster_labels = donor_labels_raw[mask].copy()
            cluster_points = self._transform_cluster(cluster_points)
            cluster_points = self._relocate_cluster(cluster_points, augmented_points)
            augmented_points = np.concatenate([augmented_points, cluster_points], axis=0)
            augmented_labels = np.concatenate([augmented_labels, cluster_labels], axis=0)
        return augmented_points, augmented_labels

    def _load_or_compute_stats(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    data = json.load(f)
                freq = np.array(data["freq"], dtype=np.int64)
                class_to_frames = {int(k): v for k, v in data["class_to_frames"].items()}
                if freq.shape[0] == self.n_classes:
                    return freq, class_to_frames
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass

        freq = np.zeros(self.n_classes, dtype=np.int64)
        class_to_frames: Dict[int, List[int]] = {c: [] for c in range(self.n_classes)}
        print(f'KimiAugmentor: computing class stats over {len(self.dataset)} frames (cache: {self.cache_path})')
        for idx in range(len(self.dataset)):
            _, labels_raw, _ = self.dataset.loadDataByIndex(idx)
            mapped = self.label_mapper(labels_raw)
            unique, counts = np.unique(mapped, return_counts=True)
            freq[unique] += counts.astype(np.int64)
            for cls_id in unique:
                class_to_frames[int(cls_id)].append(idx)

        cache_payload = {
            "freq": freq.tolist(),
            "class_to_frames": {str(k): v for k, v in class_to_frames.items()},
        }
        try:
            with open(self.cache_path, "w") as f:
                json.dump(cache_payload, f)
        except OSError:
            pass
        return freq, class_to_frames

    def _select_minor_classes(self, cls_freq: np.ndarray) -> List[int]:
        total = float(cls_freq.sum())
        if total <= 0:
            return []
        rel_freq = cls_freq / total
        minor = [
            idx for idx, f in enumerate(rel_freq)
            if f > 0 and f <= self.minor_freq_ratio and idx not in self.ignore_classes
        ]
        if self.max_minor_classes is not None and len(minor) > self.max_minor_classes:
            minor = sorted(minor, key=lambda c: rel_freq[c])[: self.max_minor_classes]
        return minor

    def _sample_minor_class(self):
        if len(self.minor_classes) == 0:
            return None
        return random.choice(self.minor_classes)

    def _sample_donor_frame(self, cls_id: int, target_idx: int):
        frames = self.class_to_frames.get(cls_id, [])
        if len(frames) == 0:
            return None
        # Avoid picking the exact same frame when possible
        candidates = [f for f in frames if f != target_idx]
        if len(candidates) == 0:
            candidates = frames
        for _ in range(self.max_donor_tries):
            donor_idx = random.choice(candidates)
            return donor_idx
        return None

    def _transform_cluster(self, cluster: np.ndarray):
        coords = cluster[:, :3]
        scale_min, scale_max = float(self.scale_range[0]), float(self.scale_range[1])
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        if scale_max > 0 and scale_max != 1.0:
            scale = random.uniform(scale_min, scale_max)
            coords *= scale

        if self.rotation_deg > 0:
            yaw = math.radians(random.uniform(-self.rotation_deg, self.rotation_deg))
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            rot_mat = np.array([[cos_yaw, -sin_yaw, 0.0],
                                [sin_yaw, cos_yaw, 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
            coords = coords @ rot_mat.T

        jitter = np.random.normal(scale=self.translation_std, size=(1, 3))
        coords = coords + jitter
        cluster[:, :3] = coords
        return cluster.astype(np.float32, copy=False)

    def _relocate_cluster(self, cluster: np.ndarray, target_points: np.ndarray):
        if target_points.shape[0] == 0:
            return cluster
        anchor_idx = random.randrange(target_points.shape[0])
        anchor_point = target_points[anchor_idx, :3]
        ground_z = np.percentile(target_points[:, 2], self.anchor_percentile)
        desired_min_z = ground_z + np.random.normal(scale=max(self.translation_std[2] * 0.5, 1e-3))

        cluster_xy_center = cluster[:, :2].mean(axis=0)
        cluster[:, 0] += anchor_point[0] - cluster_xy_center[0]
        cluster[:, 1] += anchor_point[1] - cluster_xy_center[1]

        z_shift = desired_min_z - cluster[:, 2].min()
        cluster[:, 2] += z_shift
        return cluster.astype(np.float32, copy=False)
