import argparse
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Reporter:
    def __init__(self):
        self.ok = 0
        self.warn = 0
        self.fail = 0

    def pass_(self, message: str):
        self.ok += 1
        print(f"[PASS] {message}")

    def warn_(self, message: str):
        self.warn += 1
        print(f"[WARN] {message}")

    def fail_(self, message: str):
        self.fail += 1
        print(f"[FAIL] {message}")


def resolve_path(value, base_dir: Path):
    if value in (None, "", "null"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def check_exists(rep: Reporter, path: Path, label: str, expect_dir=None):
    if path is None:
        rep.warn_(f"{label}: not set")
        return False
    if not path.exists():
        rep.fail_(f"{label}: missing -> {path}")
        return False
    if expect_dir is True and not path.is_dir():
        rep.fail_(f"{label}: expected directory -> {path}")
        return False
    if expect_dir is False and not path.is_file():
        rep.fail_(f"{label}: expected file -> {path}")
        return False
    rep.pass_(f"{label}: {path}")
    return True


def check_repo_files(rep: Reporter, repo_root: Path):
    required_files = [
        repo_root / "main.py",
        repo_root / "train.py",
        repo_root / "option.py",
        repo_root / "dataset" / "range_view_loader.py",
        repo_root / "models" / "rangevit.py",
    ]
    for path in required_files:
        check_exists(rep, path, f"repo file `{path.relative_to(repo_root)}`", expect_dir=False)


def check_semantickitti(rep: Reporter, repo_root: Path, data_root: Path, has_label: bool):
    cfg_path = repo_root / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"
    check_exists(rep, cfg_path, "SemanticKITTI label config", expect_dir=False)
    if not check_exists(rep, data_root, "SemanticKITTI data_root", expect_dir=True):
        return

    seq_dir = data_root / "00"
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    check_exists(rep, seq_dir, "SemanticKITTI sample sequence `00`", expect_dir=True)
    check_exists(rep, velodyne_dir, "SemanticKITTI velodyne dir", expect_dir=True)
    if has_label:
        check_exists(rep, labels_dir, "SemanticKITTI labels dir", expect_dir=True)

    bin_count = len(list(velodyne_dir.glob("*.bin"))) if velodyne_dir.exists() else 0
    if bin_count > 0:
        rep.pass_(f"SemanticKITTI velodyne files found: {bin_count}")
    else:
        rep.fail_("SemanticKITTI velodyne files found: 0")

    if has_label:
        label_count = len(list(labels_dir.glob("*.label"))) if labels_dir.exists() else 0
        if label_count > 0:
            rep.pass_(f"SemanticKITTI label files found: {label_count}")
        else:
            rep.fail_("SemanticKITTI label files found: 0")


def check_nuscenes(rep: Reporter, repo_root: Path, data_root: Path):
    info_path = repo_root / "dataset" / "nuScenes" / "nuscenes_lidar_n_label_data_info.json"
    check_exists(rep, info_path, "nuScenes split metadata", expect_dir=False)
    if not check_exists(rep, data_root, "nuScenes data_root", expect_dir=True):
        return

    samples_dir = data_root / "samples"
    sweeps_dir = data_root / "sweeps"
    lidarseg_dir = data_root / "lidarseg"
    check_exists(rep, samples_dir, "nuScenes samples dir", expect_dir=True)
    check_exists(rep, sweeps_dir, "nuScenes sweeps dir", expect_dir=True)
    if lidarseg_dir.exists():
        rep.pass_(f"nuScenes lidarseg dir: {lidarseg_dir}")
    else:
        rep.warn_(f"nuScenes lidarseg dir missing: {lidarseg_dir} (needed for labeled training/validation)")


def check_optional_paths(rep: Reporter, cfg: dict, config_path: Path):
    pretrained = resolve_path(cfg.get("pretrained_model"), config_path.parent)
    checkpoint = resolve_path(cfg.get("checkpoint"), config_path.parent)
    mlflow_cfg = cfg.get("mlflow", {})

    if pretrained is not None:
        check_exists(rep, pretrained, "pretrained_model", expect_dir=False)
    else:
        rep.warn_("pretrained_model: not set")

    if checkpoint is not None:
        check_exists(rep, checkpoint, "checkpoint", expect_dir=False)
    else:
        rep.warn_("checkpoint: not set")

    if mlflow_cfg.get("enable", False):
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if isinstance(tracking_uri, str) and tracking_uri.startswith("file:"):
            local_path = resolve_path(tracking_uri[len("file:"):], config_path.parent)
            if local_path is not None:
                rep.warn_(f"MLflow local artifact root should be writable: {local_path}")
        elif tracking_uri:
            rep.warn_(f"MLflow tracking URI configured: {tracking_uri}")


def check_aug_artifacts(rep: Reporter, cfg: dict, config_path: Path):
    adapted = cfg.get("adapted_augmentation", {})
    cutmix = adapted.get("instance_cutmix", {})
    if cutmix.get("enable", False):
        bank_root = resolve_path(cutmix.get("instance_bank_root", "cache/instance_bank"), config_path.parent)
        check_exists(rep, bank_root, "instance_cutmix.instance_bank_root", expect_dir=True)


def main():
    parser = argparse.ArgumentParser(description="Check that required training files exist for a given config.")
    parser.add_argument("config", help="Path to a YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    rep = Reporter()

    if not check_exists(rep, config_path, "config file", expect_dir=False):
        print("\nSummary: cannot continue without a config file.")
        sys.exit(2)

    try:
        cfg = load_yaml(config_path) or {}
        rep.pass_("config YAML parsed successfully")
    except Exception as exc:
        rep.fail_(f"config YAML parse error: {exc}")
        print(f"\nSummary: PASS={rep.ok} WARN={rep.warn} FAIL={rep.fail}")
        sys.exit(2)

    check_repo_files(rep, repo_root)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset")
    data_root = resolve_path(data_cfg.get("data_root"), config_path.parent)
    has_label = bool(data_cfg.get("has_label", True))

    if dataset_name == "SemanticKitti":
        check_semantickitti(rep, repo_root, data_root, has_label)
    elif dataset_name == "nuScenes":
        check_nuscenes(rep, repo_root, data_root)
    else:
        rep.fail_(f"unsupported or missing data.dataset: {dataset_name}")

    check_optional_paths(rep, cfg, config_path)
    check_aug_artifacts(rep, cfg, config_path)

    print(f"\nSummary: PASS={rep.ok} WARN={rep.warn} FAIL={rep.fail}")
    sys.exit(1 if rep.fail else 0)


if __name__ == "__main__":
    main()
