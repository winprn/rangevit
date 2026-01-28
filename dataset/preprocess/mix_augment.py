import os
import numpy as np


def _rotate_points_z(points, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array(
        [[cos_a, -sin_a, 0.0],
         [sin_a, cos_a, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=points.dtype,
    )
    out = points.copy()
    out[:, :3] = out[:, :3] @ rot.T
    return out


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class ClusterMix(object):
    def __init__(self, prob=0.5, vertical_prob=0.5, sector_width=np.pi / 2, height_ratio=0.5):
        self.prob = prob
        self.vertical_prob = vertical_prob
        self.sector_width = sector_width
        self.height_ratio = height_ratio

    def __call__(self, pcloud, labels, mix_points, mix_labels):
        if np.random.rand() > self.prob:
            return pcloud, labels

        if np.random.rand() < self.vertical_prob:
            yaw = np.arctan2(pcloud[:, 1], pcloud[:, 0])
            mix_yaw = np.arctan2(mix_points[:, 1], mix_points[:, 0])
            center = np.random.uniform(-np.pi, np.pi)
            mask = _angle_diff(yaw, center) <= (self.sector_width * 0.5)
            mix_mask = _angle_diff(mix_yaw, center) <= (self.sector_width * 0.5)
        else:
            z_thresh = np.quantile(pcloud[:, 2], self.height_ratio)
            mask = pcloud[:, 2] > z_thresh
            mix_mask = mix_points[:, 2] > z_thresh

        keep = ~mask
        points = np.concatenate([pcloud[keep], mix_points[mix_mask]], axis=0)
        labels = np.concatenate([labels[keep], mix_labels[mix_mask]], axis=0)
        return points, labels


class InstanceCopy(object):
    def __init__(self, prob=0.5, classes=None, max_instances_per_class=1):
        self.prob = prob
        self.classes = classes
        self.max_instances_per_class = max_instances_per_class

    def __call__(self, pcloud, labels, mix_points, mix_labels, mix_instances=None):
        if np.random.rand() > self.prob:
            return pcloud, labels
        if mix_instances is None:
            return pcloud, labels

        classes = self.classes
        if classes is None:
            classes = np.unique(mix_labels)

        added_points = [pcloud]
        added_labels = [labels]

        for cls_id in classes:
            where_cls = mix_labels == cls_id
            if where_cls.sum() == 0:
                continue
            inst_ids = np.unique(mix_instances[where_cls])
            inst_ids = inst_ids[inst_ids > 0]
            if inst_ids.size == 0:
                continue
            num_to_copy = min(self.max_instances_per_class, inst_ids.size)
            chosen = np.random.choice(inst_ids, num_to_copy, replace=False)
            for inst_id in chosen:
                inst_mask = (mix_instances == inst_id) & where_cls
                if inst_mask.sum() == 0:
                    continue
                added_points.append(mix_points[inst_mask])
                added_labels.append(mix_labels[inst_mask])

        points = np.concatenate(added_points, axis=0)
        labels = np.concatenate(added_labels, axis=0)
        return points, labels


class PolarMix(object):
    def __init__(self, classes=None, prob=0.5):
        self.classes = classes or []
        self.prob = prob

    def __call__(self, pc1, label1, pc2, label2):
        if np.random.rand() < self.prob:
            sector = np.random.uniform(-np.pi, np.pi)
            theta1 = (np.arctan2(pc1[:, 1], pc1[:, 0]) - sector) % (2 * np.pi)
            mask1 = ~((theta1 > 0) & (theta1 < np.pi))
            theta2 = (np.arctan2(pc2[:, 1], pc2[:, 0]) - sector) % (2 * np.pi)
            mask2 = (theta2 > 0) & (theta2 < np.pi)
            pc = np.concatenate((pc1[mask1], pc2[mask2]), axis=0)
            label = np.concatenate((label1[mask1], label2[mask2]), axis=0)
        else:
            pc = pc1
            label = label1

        if self.classes:
            cls_mask = label2 == self.classes[0]
            for cls_id in self.classes[1:]:
                cls_mask |= label2 == cls_id
            if cls_mask.sum() > 0:
                pc2_sel = pc2[cls_mask]
                label2_sel = label2[cls_mask]
                pc2_rot1 = _rotate_points_z(pc2_sel, np.random.uniform(-np.pi, np.pi))
                pc2_rot2 = _rotate_points_z(pc2_sel, np.random.uniform(-np.pi, np.pi))
                pc = np.concatenate((pc, pc2_sel, pc2_rot1, pc2_rot2), axis=0)
                label = np.concatenate((label, label2_sel, label2_sel, label2_sel), axis=0)

        return pc, label


class InstanceCutMix(object):
    def __init__(self, rootdir, classes, num_to_add=(0, 2), min_points=20, prob=1.0):
        self.rootdir = rootdir
        self.classes = classes
        self.num_to_add = num_to_add
        self.min_points = min_points
        self.prob = prob
        self.bank = {}
        self.__loaded__ = False

        os.makedirs(self.rootdir, exist_ok=True)
        for cls_id in classes:
            cls_dir = os.path.join(self.rootdir, str(cls_id))
            os.makedirs(cls_dir, exist_ok=True)
            self.bank[cls_id] = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".bin")
            ]
        self.__loaded__ = all(len(self.bank[k]) > 0 for k in self.bank.keys())

    def cut(self, pc, class_label, instance_label):
        for cls_id in self.classes:
            where_class = class_label == cls_id
            inst_ids = np.unique(instance_label[where_class])
            for inst_id in inst_ids:
                inst_mask = instance_label == inst_id
                if inst_mask.sum() < self.min_points:
                    continue
                inst_points = pc[inst_mask].copy()
                inst_points[:, :2] -= inst_points[:, :2].mean(0, keepdims=True)
                inst_points[:, 2] -= inst_points[:, 2].min(0, keepdims=True)
                cls_dir = os.path.join(self.rootdir, str(cls_id))
                out_path = os.path.join(cls_dir, f"{len(self.bank[cls_id]):07d}.bin")
                inst_points.astype(np.float32).tofile(out_path)
                self.bank[cls_id].append(out_path)

    def mix(self, pc, class_label):
        new_pc = [pc]
        new_label = [class_label]

        for cls_id in self.classes:
            bank_list = self.bank.get(cls_id, [])
            if len(bank_list) == 0:
                continue
            num_min, num_max = self.num_to_add
            if num_max < num_min:
                num_max = num_min
            num_to_add = np.random.randint(num_min, num_max + 1)
            if num_to_add <= 0:
                continue
            choices = np.random.choice(len(bank_list), num_to_add, replace=True)
            for idx in choices:
                inst = np.fromfile(bank_list[idx], dtype=np.float32).reshape((-1, 4))
                inst = _rotate_points_z(inst, np.random.uniform(-np.pi, np.pi))
                anchor = pc[np.random.randint(0, pc.shape[0])]
                inst[:, :3] += anchor[:3][None, :]
                new_pc.append(inst)
                new_label.append(np.full((inst.shape[0],), cls_id, dtype=class_label.dtype))

        return np.concatenate(new_pc, axis=0), np.concatenate(new_label, axis=0)

    def __call__(self, pc, class_label, instance_label):
        if np.random.rand() > self.prob:
            return pc, class_label
        if not self.__loaded__:
            self.cut(pc, class_label, instance_label)
            self.__loaded__ = all(len(self.bank[k]) > 0 for k in self.bank.keys())
            return pc, class_label
        return self.mix(pc, class_label)
