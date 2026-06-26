# Copyright 2022 - Valeo Comfort and Driving Assistance
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

import json
import os

import yaml
import numpy as np
from PIL import Image


# BALViT-style skip ratio -> JSON key mapping. NOTE: BALViT's original
# naming uses keys like "0.1" / "0.01" / "0.001" which look like percent
# values but actually correspond to ``skip_ratio`` values 10 / 100 / 1000
# (so they represent 10%, 1%, 0.1% of the data respectively). We use the
# clearer percent-based labels here.
SKIP_RATIO_TO_PERCENT = {
    10: "10pct",
    100: "1pct",
    1000: "0.1pct",
}

DEFAULT_PERCENTILES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "percentiles_split.json",
)


def _rewrite_root(path: str, original_prefix: str, new_root: str) -> str:
    """Replace the dataset root prefix in a stored path so the percentiles
    JSON (generated against one root) works for any other root.
    """
    norm = path.replace("\\", "/")
    if original_prefix and norm.startswith(original_prefix.replace("\\", "/")):
        suffix = norm[len(original_prefix.replace("\\", "/")):]
        suffix = suffix.lstrip("/")
        return os.path.join(new_root, suffix)
    return os.path.join(new_root, os.path.basename(path))


class SemanticKitti(object):
    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_label=True,
                 split='train',
                 skip_ratio=1,
                 percentiles_json=DEFAULT_PERCENTILES_JSON):
        self.root = root
        self.sequences = sequences
        self.sequences.sort()  # sort seq id
        self.has_label = has_label
        self.split = split
        self.skip_ratio = int(skip_ratio) if skip_ratio is not None else 1
        self.percentiles_json = percentiles_json

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, 'r'))
        else:
            raise ValueError(f'Config file not found: {config_path}')

        if os.path.isdir(self.root):
            print(f'Dataset found: {self.root}')
        else:
            raise ValueError(f'Dataset not found: {self.root}')

        self.pointcloud_files = []
        self.label_files = []
        for seq in self.sequences:
            # format seq id
            seq = '{0:02d}'.format(int(seq))
            print(f'parsing seq {seq}...')

            # get file list from path
            pointcloud_path = os.path.join(self.root, seq, 'velodyne')
            pointcloud_files = [
                os.path.join(pointcloud_path, f)
                for f in os.listdir(pointcloud_path) if '.bin' in f]

            if self.has_label:
                label_path = os.path.join(self.root, seq, 'labels')
                label_files = [
                    os.path.join(label_path, f)
                    for f in os.listdir(label_path) if '.label' in f]

            if self.has_label:
                assert (len(pointcloud_files) == len(label_files))

            self.pointcloud_files.extend(pointcloud_files)
            if self.has_label:
                self.label_files.extend(label_files)

        # sort for correspondance
        self.pointcloud_files.sort()
        if self.has_label:
            self.label_files.sort()

        if (
            self.skip_ratio is not None
            and self.skip_ratio > 1
            and self.split == "train"
        ):
            if self.skip_ratio not in SKIP_RATIO_TO_PERCENT:
                raise ValueError(
                    f"Unsupported skip_ratio {self.skip_ratio}. Supported values "
                    f"are {sorted(SKIP_RATIO_TO_PERCENT)}."
                )
            percentage = SKIP_RATIO_TO_PERCENT[self.skip_ratio]
            if not os.path.isfile(self.percentiles_json):
                raise FileNotFoundError(
                    f"Percentiles split file not found: {self.percentiles_json}. "
                    "Run scripts/generate_percentiles_split.py to create it."
                )
            with open(self.percentiles_json, "r") as p:
                splits = json.load(p)
            if percentage not in splits:
                raise ValueError(
                    f"Percentage {percentage!r} missing from "
                    f"{self.percentiles_json}. Available: {sorted(splits)}."
                )

            self.pointcloud_files = []
            self.label_files = []
            for seq, paths in splits[percentage].items():
                self.pointcloud_files.extend(paths.get("points", []))
                self.label_files.extend(paths.get("labels", []))

            # The percentiles JSON stores paths generated against a specific
            # dataset root (typically a placeholder). Rewrite the root so
            # the same JSON works regardless of where the dataset lives.
            placeholder_root = (
                "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences"
            )
            self.pointcloud_files = [
                _rewrite_root(p, placeholder_root, self.root)
                for p in self.pointcloud_files
            ]
            self.label_files = [
                _rewrite_root(p, placeholder_root, self.root)
                for p in self.label_files
            ]
            print(
                f"Applied BALViT percentile split {percentage} "
                f"(skip_ratio={self.skip_ratio}): "
                f"{len(self.pointcloud_files)} pointclouds selected from "
                f"{self.percentiles_json}"
            )

        print(f'Using {len(self.pointcloud_files)} pointclouds from sequences {self.sequences}')

        # load config -------------------------------------
        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config['learning_map']
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config['learning_map_inv']
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = self.data_config['content']
        content = np.zeros(len(self.data_config['learning_map_inv']), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config['mapped_class_name']

    @staticmethod
    def readPCD(path):
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pcd

    @staticmethod
    def readLabel(path):
        label = np.fromfile(path, dtype=np.int32)
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        return sem_label, inst_label

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if '\\' in path:
            # windows path
            path_split = path.split('\\')
        else:
            path_split = path.split('/')
        seq_id = path_split[-3]
        frame_id = path_split[-1].split('.')[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByIndex(self, index):
        sem_label, inst_label = self.readLabel(self.label_files[index])
        return sem_label, inst_label

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem_label, inst_label = self.readLabel(self.label_files[index])
        else:
            sem_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
            inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label

    def __len__(self):
        return len(self.pointcloud_files)
