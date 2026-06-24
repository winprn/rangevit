"""Generate ``dataset/semantic_kitti/percentiles_split.json``.

This reproduces the JSON used by the reference BALViT
(``BALViT/balvit/dataset/semantic_kitti/parser.py``) so that label-efficient
training in RangeTinyVim matches the BALViT protocol exactly.

Schema (matching BALViT)::

    {
      "0.1":  { "<seq_id>": {"points": [...], "labels": [...]}, ... },
      "0.01": { ... },
      "0.001":{ ... }
    }

For each percentage and each train sequence:
1. Sort filenames lexicographically (matches the on-disk ordering).
2. Pick every N-th scan (``step = max(1, round(seq_len / target_count))``).
3. Store both the velodyne and label paths.

The script must be run once after the dataset is downloaded; the resulting
JSON is committed to the repo so training is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List

import yaml


# Labels for the JSON keys. These match the parser's SKIP_RATIO_TO_PERCENT
# in dataset/semantic_kitti/parser.py and the YAML filenames.
# NOTE on the BALViT convention: BALViT's original skip_to_percent uses keys
# "0.1", "0.01", "0.001" which look like percentages but actually correspond
# to skip_ratio=10/100/1000 (i.e. 10%/1%/0.1% of the data). We use the
# clearer "10pct" / "1pct" / "0.1pct" labels that match the actual fraction.
SKIP_RATIO_TO_PERCENT = {
    10: "10pct",
    100: "1pct",
    1000: "0.1pct",
}

# Canonical SemanticKITTI train sequences (mirrors
# dataset/semantic_kitti/semantic-kitti.yaml -> split.train).
DEFAULT_TRAIN_SEQUENCES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


def collect_sequence_scans(
    data_root: str,
    seq_id: int,
    has_label: bool = True,
    placeholder_root: str = "Datasets/SemanticKitti/dataset/sequences",
) -> Dict[str, List[str]]:
    """Return the lexicographically sorted list of velodyne / label paths for
    a single sequence, normalized against ``placeholder_root`` so the
    generated JSON is portable.
    """
    seq_str = f"{int(seq_id):02d}"
    velo_dir = os.path.join(data_root, seq_str, "velodyne")
    label_dir = os.path.join(data_root, seq_str, "labels")

    def _normalize(p: str) -> str:
        # Preserve everything after the sequence id (e.g. "velodyne/000000.bin")
        # so the JSON path can be re-rooted to the user's data root and the
        # file will actually exist on disk.
        suffix = p.replace("\\", "/")
        marker = f"/{seq_str}/"
        if marker in suffix:
            suffix = suffix.split(marker, 1)[1]
        return os.path.join(placeholder_root, seq_str, suffix).replace("\\", "/")

    points: List[str] = []
    labels: List[str] = []

    if os.path.isdir(velo_dir):
        point_files = sorted(
            f for f in os.listdir(velo_dir) if f.endswith(".bin")
        )
        points = [_normalize(os.path.join(velo_dir, f)) for f in point_files]
        if has_label and os.path.isdir(label_dir):
            label_files = sorted(
                f for f in os.listdir(label_dir) if f.endswith(".label")
            )
            labels = [_normalize(os.path.join(label_dir, f)) for f in label_files]

    return {"points": points, "labels": labels}


def subsample_sequence(
    points: List[str],
    labels: List[str],
    target_count: int,
) -> Dict[str, List[str]]:
    """Pick every N-th scan so the resulting list has approximately
    ``target_count`` entries.
    """
    if not points:
        return {"points": [], "labels": []}

    seq_len = len(points)
    if target_count <= 0 or target_count >= seq_len:
        return {"points": list(points), "labels": list(labels)}

    step = max(1, round(seq_len / target_count))
    keep = list(range(0, seq_len, step))

    out_points = [points[i] for i in keep]
    out_labels = [labels[i] for i in keep] if labels else []
    return {"points": out_points, "labels": out_labels}


def count_total_scans(per_seq_scans: Dict[str, Dict[str, List[str]]]) -> int:
    return sum(len(v["points"]) for v in per_seq_scans.values())


def build_splits(
    data_root: str,
    sequences: List[int],
    has_label: bool,
    skip_ratios: List[int],
) -> "OrderedDict[str, OrderedDict[str, Dict[str, List[str]]]]":
    """Walk every train sequence once, then derive each percentage's split
    by deterministic subsampling.
    """
    per_seq_scans: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict()
    for seq_id in sequences:
        seq_str = f"{int(seq_id):02d}"
        per_seq_scans[seq_str] = collect_sequence_scans(
            data_root, seq_id, has_label=has_label
        )

    total_scans = count_total_scans(per_seq_scans)
    print(
        f"[generate_percentiles_split] Discovered {total_scans} scans across "
        f"{len(per_seq_scans)} sequences (root={data_root})."
    )

    splits: "OrderedDict[str, OrderedDict[str, Dict[str, List[str]]]]" = (
        OrderedDict()
    )
    for skip_ratio in skip_ratios:
        percent = SKIP_RATIO_TO_PERCENT[skip_ratio]
        target_count = max(1, int(round(total_scans / skip_ratio)))
        per_seq: "OrderedDict[str, Dict[str, List[str]]]" = OrderedDict()
        running_total = 0
        for seq_str, scans in per_seq_scans.items():
            seq_target = max(
                1, int(round(len(scans["points"]) / skip_ratio))
            )
            sampled = subsample_sequence(
                scans["points"], scans["labels"], seq_target
            )
            per_seq[seq_str] = sampled
            running_total += len(sampled["points"])
        splits[percent] = per_seq
        print(
            f"[generate_percentiles_split] skip_ratio={skip_ratio} -> "
            f"{percent}% selected {running_total} / {total_scans} scans "
            f"(target {target_count})."
        )

    return splits


def write_json(splits: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[generate_percentiles_split] Wrote {output_path}")


def resolve_train_sequences(
    data_root: str, sequences: List[int] | None
) -> List[int]:
    """Return the user-supplied sequence list, or fall back to the canonical
    SemanticKITTI train split when ``sequences`` is None.
    """
    if sequences:
        return [int(s) for s in sequences]
    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "dataset",
        "semantic_kitti",
        "semantic-kitti.yaml",
    )
    yaml_path = os.path.normpath(yaml_path)
    if os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        seqs = (cfg or {}).get("split", {}).get("train")
        if seqs:
            return [int(s) for s in seqs]
    return list(DEFAULT_TRAIN_SEQUENCES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dataset/semantic_kitti/percentiles_split.json for "
            "BALViT-compatible label-efficient training."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=(
            "../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences"
        ),
        help="Root directory containing sequence folders (00/, 01/, ...).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "dataset",
                "semantic_kitti",
                "percentiles_split.json",
            )
        ),
        help="Where to write the JSON file.",
    )
    parser.add_argument(
        "--no_label",
        action="store_true",
        help="Disable reading the labels/ folder.",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of train sequence ids. Defaults to the "
            "sequences declared in dataset/semantic_kitti/semantic-kitti.yaml."
        ),
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    sequences = resolve_train_sequences(data_root, args.sequences)

    if not os.path.isdir(data_root):
        print(
            f"[generate_percentiles_split] WARNING: data_root {data_root} "
            "does not exist. Writing an empty JSON scaffold; rerun this "
            "script once the dataset is downloaded."
        )

    splits = build_splits(
        data_root=data_root,
        sequences=sequences,
        has_label=not args.no_label,
        skip_ratios=[10, 100, 1000],
    )
    write_json(splits, args.output)


if __name__ == "__main__":
    main()
