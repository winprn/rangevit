import os
import yaml
import numpy as np


class SemanticPOSS(object):
    """SemanticPOSS dataset parser.

    Layout: <root>/<seq>/velodyne/*.bin, labels/*.label, tag/*.tag.
    Tag files are H*W boolean masks identifying which range-image cells
    a 1-to-1 point->cell projection occupies.
    """

    H = 40
    W = 1800

    @staticmethod
    def _frame_stems(paths):
        return [os.path.splitext(os.path.basename(path))[0] for path in paths]

    def __init__(self, root, sequences, config_path, has_label=True):
        self.root = root
        self.sequences = sorted(int(s) for s in sequences)
        self.has_label = has_label

        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        self.data_config = yaml.safe_load(open(config_path, "r"))

        if not os.path.isdir(self.root):
            raise ValueError(f"Dataset not found: {self.root}")

        self.pointcloud_files = []
        self.label_files = []
        self.tag_files = []
        for seq in self.sequences:
            seq_str = f"{seq:02d}"
            seq_dir = os.path.join(self.root, seq_str)
            pc_dir = os.path.join(seq_dir, "velodyne")
            tag_dir = os.path.join(seq_dir, "tag")
            pc_files = sorted(
                os.path.join(pc_dir, f) for f in os.listdir(pc_dir) if f.endswith(".bin")
            )
            tag_files = sorted(
                os.path.join(tag_dir, f) for f in os.listdir(tag_dir) if f.endswith(".tag")
            )
            assert len(pc_files) == len(tag_files), (
                f"Seq {seq_str}: {len(pc_files)} bins vs {len(tag_files)} tags"
            )
            pc_stems = self._frame_stems(pc_files)
            tag_stems = self._frame_stems(tag_files)
            if pc_stems != tag_stems:
                raise ValueError(
                    f"Seq {seq_str}: point cloud and tag frame ids do not match"
                )

            if self.has_label:
                lbl_dir = os.path.join(seq_dir, "labels")
                lbl_files = sorted(
                    os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(".label")
                )
                assert len(pc_files) == len(lbl_files), (
                    f"Seq {seq_str}: {len(pc_files)} bins vs {len(lbl_files)} labels"
                )
                label_stems = self._frame_stems(lbl_files)
                if pc_stems != label_stems:
                    raise ValueError(
                        f"Seq {seq_str}: point cloud and label frame ids do not match"
                    )
                self.label_files.extend(lbl_files)

            self.pointcloud_files.extend(pc_files)
            self.tag_files.extend(tag_files)

        print(f"Using {len(self.pointcloud_files)} POSS frames from sequences {self.sequences}")

        learning_map = self.data_config["learning_map"]
        max_key = max(learning_map.keys())
        self.class_map_lut = np.zeros((max_key + 100,), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v

        learning_map_inv = self.data_config["learning_map_inv"]
        max_inv = max(learning_map_inv.keys())
        self.class_map_lut_inv = np.zeros((max_inv + 100,), dtype=np.int32)
        for k, v in learning_map_inv.items():
            self.class_map_lut_inv[k] = v

        cls_content = self.data_config["content"]
        n_mapped = len(self.data_config["learning_map_inv"])
        content = np.zeros(n_mapped, dtype=np.float32)
        for raw_id, freq in cls_content.items():
            content[self.class_map_lut[raw_id]] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    @staticmethod
    def readPCD(path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def readLabel(path):
        raw = np.fromfile(path, dtype=np.uint32)
        sem = (raw & 0xFFFF).astype(np.int32)
        inst = (raw >> 16).astype(np.int32)
        return sem, inst

    @classmethod
    def readTag(cls, path):
        # Files are written as np.bool_ (1 byte). Read as uint8 then cast.
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != cls.H * cls.W:
            raise ValueError(
                f"Tag length {raw.size} != expected {cls.H * cls.W} for {path}"
            )
        return raw.astype(np.bool_)

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        parts = path.replace("\\", "/").split("/")
        seq_id = parts[-3]
        frame_id = parts[-1].split(".")[0]
        return seq_id, frame_id

    def labelMapping(self, label):
        return self.class_map_lut[label]

    def loadLabelByIndex(self, index):
        return self.readLabel(self.label_files[index])

    def loadDataByIndex(self, index):
        pc = self.readPCD(self.pointcloud_files[index])
        if self.has_label:
            sem, inst = self.readLabel(self.label_files[index])
        else:
            sem = np.zeros(pc.shape[0], dtype=np.int32)
            inst = np.zeros(pc.shape[0], dtype=np.int32)
        return pc, sem, inst

    def loadTagByIndex(self, index):
        if index < 0 or index >= len(self.tag_files):
            raise IndexError(
                f"Tag index {index} out of range for {len(self.tag_files)} tag files"
            )
        return self.readTag(self.tag_files[index])

    def __len__(self):
        return len(self.pointcloud_files)
