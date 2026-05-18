"""Range image-level augmentations for LiDAR semantic segmentation.

Operates on batched GPU tensors (B, C, H, W) after range projection.
Pairs samples within the batch via derangement and applies mixing-based
augmentations that improve tail-class representation and scene diversity.

Ported from RangeRet (dataloader/rangeaug.py).
"""

import torch
import random


def match_elements(n):
    """Generate a derangement of [0..n-1]: a permutation where no element maps to itself.

    Returns a dict mapping each index to its paired index.
    """
    list1 = list(range(n))
    finished = False
    while not finished:
        try:
            list2 = list1.copy()
            random.shuffle(list2)
            matches = {}
            not_allowed = set()
            for i in range(n):
                current_element = list1[i]
                for j in range(n):
                    if list2[j] != current_element and (current_element, list2[j]) not in not_allowed:
                        matches[current_element] = list2[j]
                        not_allowed.add((current_element, list2[j]))
                        del list2[j]
                        break
            finished = True
        except:
            finished = False

    return matches


class RangeAugmentation:
    """Applies range image-level augmentations on a batch of projected LiDAR scans.

    Each sample in the batch is paired with a different sample (via derangement),
    then five augmentation techniques are applied sequentially with independent
    probability rolls:
      - RangePolar:      mix azimuth columns + flip tail-class instances
      - RangeBeams:      mix horizontal laser beam bands
      - RangeCompletion: fill void pixels by row-shifting
      - RangeFake:       relabel front-class as tail-class (disabled by default)
      - RangeInstance:   paste tail-class pixels from paired sample

    Args:
        aug_prob: Per-technique probabilities, ordered
            [RangePolar, RangeBeams, RangeCompletion, RangeFake, RangeInstance].
            Default: [0.9, 0.7, 0.9, 0.0, 0.9].
        tail_classes: Mapped class ids treated as tail (underrepresented).
            Defaults to the SemanticKITTI tail set.
        fake_pairs: Mapping from a front-class id to a list of tail-class ids
            it may be relabelled as during RangeFake. Defaults to the
            SemanticKITTI pair map. RangeFake is off by default; if turned on
            for a non-KITTI dataset this map MUST be overridden.
    """

    _DEFAULT_TAIL_CLASSES = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]
    _DEFAULT_FAKE_PAIRS = {1: [2, 3, 5, 8], 9: [10, 11, 12]}

    def __init__(self, aug_prob=None, tail_classes=None, fake_pairs=None):
        if aug_prob is None:
            aug_prob = [0.9, 0.7, 0.9, 0.0, 0.9]
        self.aug_prob = aug_prob
        self.tail_classes = list(tail_classes) if tail_classes is not None else list(self._DEFAULT_TAIL_CLASSES)
        self.fake_pairs = dict(fake_pairs) if fake_pairs is not None else dict(self._DEFAULT_FAKE_PAIRS)
        print(
            f'[INFO] Range image augmentation enabled with probabilities {aug_prob}, '
            f'tail_classes={self.tail_classes}'
        )

    def __call__(self, data, label, mask):
        """Apply range augmentations to a batch.

        Args:
            data:  (B, C, H, W) float tensor — range image features.
            label: (B, H, W) long tensor — semantic labels.
            mask:  (B, H, W) float tensor — valid-pixel mask (1=valid, 0=void).

        Returns:
            Tuple of (augmented_data, augmented_label) with same shapes.
        """
        B = data.shape[0]
        if B < 2:
            return data, label

        out_scan = []
        out_label = []
        match_dict = match_elements(B)

        for i in range(B):
            j = match_dict[i]

            scan_a, scan_b = data[i].clone(), data[j]
            label_a, label_b = label[i].clone(), label[j]
            mask_a = mask[i].clone()

            if torch.rand(1) < self.aug_prob[0]:
                scan_a, label_a = self._range_polar(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[1]:
                scan_a, label_a = self._range_beams(scan_a, label_a, scan_b, label_b)
            if torch.rand(1) < self.aug_prob[2]:
                scan_a, label_a = self._range_completion(scan_a, label_a, mask_a)
            if torch.rand(1) < self.aug_prob[3]:
                scan_a, label_a = self._range_fake(scan_a, label_a)
            if torch.rand(1) < self.aug_prob[4]:
                scan_a, label_a = self._range_instance(scan_a, label_a, scan_b, label_b)

            out_scan.append(scan_a)
            out_label.append(label_a)

        out_scan = torch.stack(out_scan)
        out_label = torch.stack(out_label)

        return out_scan, out_label.long()

    def _range_polar(self, scan_a, label_a, scan_b, label_b):
        """Mix a random azimuth column range, then flip tail-class instances."""
        _, H, W = scan_a.shape
        p_start = random.randint(0, int(0.5 * W))
        p_end = random.randint(int(0.5 * W), W)
        scan_a[:, :, p_start:p_end] = scan_b[:, :, p_start:p_end]
        label_a[:, p_start:p_end] = label_b[:, p_start:p_end]

        # Extract tail-class pixels, flip horizontally, paste back
        tail_tensor = torch.tensor(self.tail_classes, device=label_a.device)
        class_mask = torch.isin(label_a, tail_tensor)
        masked_scan_a = torch.full_like(scan_a, -1.0)
        masked_scan_a[:, class_mask] = scan_a[:, class_mask]
        masked_label_a = torch.full_like(label_a, -1)
        masked_label_a[class_mask] = label_a[class_mask]

        class_mask = torch.flip(class_mask, dims=[1])
        masked_scan_a = torch.flip(masked_scan_a, dims=[2])
        masked_label_a = torch.flip(masked_label_a, dims=[1])

        scan_a[:, class_mask] = masked_scan_a[:, class_mask]
        label_a[class_mask] = masked_label_a[class_mask]
        return scan_a, label_a

    def _range_beams(self, scan_a, label_a, scan_b, label_b):
        """Mix alternating horizontal beam bands from a paired sample."""
        _, H, W = scan_a.shape
        h_mix = random.choice([2, 3, 4])
        h_step = int(H / h_mix)
        h_index = list(range(0, H, h_step))
        for i in range(len(h_index)):
            if i % 2 == 1:
                h_s = h_index[i]
                h_e = h_index[i] + h_step if h_index[i] + h_step < H else H
                scan_a[:, h_s:h_e, :] = scan_b[:, h_s:h_e, :]
                label_a[h_s:h_e, :] = label_b[h_s:h_e, :]
        return scan_a, label_a

    def _range_completion(self, scan_a, label_a, mask):
        """Fill void pixels by shifting the scan one row down."""
        shifted_scan = torch.zeros_like(scan_a)
        shifted_label = torch.zeros_like(label_a)
        shifted_scan[:, 1:, :] = scan_a[:, :-1, :]
        shifted_label[1:, :] = label_a[:-1, :]

        void = mask == 0
        scan_a[:, void] = shifted_scan[:, void]
        label_a[void] = shifted_label[void]
        return scan_a, label_a

    def _range_fake(self, scan_a, label_a):
        """Relabel front-class pixels as tail-class (unstable, disabled by default).

        Uses self.fake_pairs (front_cls -> list of tail_cls). For non-KITTI
        datasets the caller must provide an appropriate fake_pairs map.
        """
        if not self.fake_pairs:
            return scan_a, label_a
        rand_front_class = random.choice(list(self.fake_pairs.keys()))
        candidates = self.fake_pairs[rand_front_class]
        if not candidates:
            return scan_a, label_a
        rand_tail_class = random.choice(candidates)
        label_a[label_a == rand_front_class] = rand_tail_class
        return scan_a, label_a

    def _range_instance(self, scan_a, label_a, scan_b, label_b):
        """Paste all tail-class pixels from paired sample."""
        for cls in self.tail_classes:
            pix = label_b == cls
            scan_a[:, pix] = scan_b[:, pix]
            label_a[pix] = label_b[pix]
        return scan_a, label_a
