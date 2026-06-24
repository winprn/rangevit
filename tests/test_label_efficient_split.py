"""Smoke test for the BALViT label-efficient rewiring.

This lightweight version avoids importing the full project (which pulls in
torch via utils.metrics) and instead exercises the parser and YAMLs
directly, mirroring how ``train.py`` wires them together.

Run from the repo root: ``python tests/test_label_efficient_split.py``.

Prints the total dataset size and the actual fraction each YAML selects so
the relationship between ``skip_ratio`` and data percentage is unambiguous.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml  # noqa: E402


_PARSER_PATH = os.path.join(REPO_ROOT, "dataset", "semantic_kitti", "parser.py")
_spec = importlib.util.spec_from_file_location("sk_parser", _PARSER_PATH)
sk_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sk_parser)


DATASET_ROOT = os.environ.get(
    "KITTI_DATA_ROOT",
    "E:/KLTN/dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences",
)


# (yaml_relpath, dataset_skip_step_org, dataset_skip_step, repeat_factor,
#  expected_scans, expected_epoch_length, expected_fraction_label)
CONFIGS = [
    ("config/kitti/label_efficient/config_tinyvim_aug_le_10.yaml",   10,    1, 1,  1922, 1922, "10%"),
    ("config/kitti/label_efficient/config_tinyvim_aug_le_1.yaml",    100,   1, 1,  200,  200,  "1%"),
    ("config/kitti/label_efficient/config_tinyvim_aug_le_0_1.yaml", 1000,  1, 10, 24,   240,  "0.1%"),
]


def _load_yaml(relpath: str):
    path = os.path.join(REPO_ROOT, relpath)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _train_sequences():
    train_yaml = "dataset/semantic_kitti/semantic-kitti.yaml"
    cfg = yaml.safe_load(open(train_yaml, "r"))
    return cfg["split"]["train"]


def _build_parser(dataset_root: str, skip_ratio: int):
    train_yaml = "dataset/semantic_kitti/semantic-kitti.yaml"
    return sk_parser.SemanticKitti(
        root=dataset_root,
        sequences=_train_sequences(),
        config_path=train_yaml,
        split="train",
        skip_ratio=skip_ratio,
    )


def _total_train_scans() -> int:
    total = 0
    for seq in ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]:
        d = os.path.join(DATASET_ROOT, seq, "velodyne")
        total += len([f for f in os.listdir(d) if f.endswith(".bin")])
    return total


def main() -> int:
    total = _total_train_scans()
    print(f"Total SemanticKITTI train scans: {total}\n")

    failures = []
    for relpath, org, skip, repeat, expected, expected_epoch, fraction in CONFIGS:
        print(f"=== {relpath} ({fraction}) ===")
        cfg = _load_yaml(relpath)
        data = cfg.get("data", {})
        if data.get("dataset_skip_step_org") != org:
            failures.append((relpath, f"dataset_skip_step_org={data.get('dataset_skip_step_org')}, expected {org}"))
        if data.get("dataset_skip_step") != skip:
            failures.append((relpath, f"dataset_skip_step={data.get('dataset_skip_step')}, expected {skip}"))
        if data.get("repeat_factor") != repeat:
            failures.append((relpath, f"repeat_factor={data.get('repeat_factor')}, expected {repeat}"))
        if not data.get("label_efficient_enable", False):
            failures.append((relpath, "label_efficient_enable should be True"))
        try:
            trainset = _build_parser(DATASET_ROOT, org)
        except Exception as exc:
            failures.append((relpath, f"Parser construction failed: {exc}"))
            print(f"  FAIL parser: {exc}")
            continue
        n = len(trainset)
        effective = (n // max(1, skip)) * max(1, repeat)
        actual_pct = 100.0 * n / max(1, total)
        status = "OK" if n == expected else f"MISMATCH (got {n})"
        print(
            f"  parser scans: {n} (expected {expected}) -> {status} "
            f"[{actual_pct:.2f}% of {total}]"
        )
        print(f"  effective epoch length: {effective} (expected {expected_epoch})")
        if n != expected:
            failures.append((relpath, f"scan count {n} != expected {expected}"))
        if effective != expected_epoch:
            failures.append((relpath, f"effective epoch length {effective} != expected {expected_epoch}"))
        print()

    json_path = os.path.join(REPO_ROOT, "dataset", "semantic_kitti", "percentiles_split.json")
    if not os.path.isfile(json_path):
        failures.append(("percentiles_split.json", f"missing: {json_path}"))
    else:
        with open(json_path, "r") as f:
            splits = json.load(f)
        print("=== percentiles_split.json contents ===")
        for key in splits:
            seq_total = sum(len(v["points"]) for v in splits[key].values())
            pct = 100.0 * seq_total / max(1, total)
            print(
                f"  key {key!r}: {len(splits[key])} sequences, "
                f"{seq_total} scans ({pct:.2f}% of total)"
            )

    print("\n=== Summary ===")
    if failures:
        for cfg, msg in failures:
            print(f"  FAIL {cfg}: {msg}")
        return 1
    print("  All label-efficient configs produced the expected scan counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())