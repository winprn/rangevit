import argparse
import math
import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dataset.semantic_kitti import SemanticKitti
from dataset.preprocess import augmentor, projection, ClusterMix, InstanceCopy, PolarMix, InstanceCutMix
from dataset.range_view_loader import crop_inputs


DEFAULT_DATA_CONFIG = REPO_ROOT / "dataset" / "semantic_kitti" / "semantic-kitti.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifact" / "range_augmentation_preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview how the configured range-view augmentations change a SemanticKITTI frame. "
            "Exports the original range image plus one output per augmentation method."
        )
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config_kitti_tinyvim.yaml"))
    parser.add_argument("--data-config", type=str, default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=7, help="Random seed for augmentation sampling.")
    parser.add_argument(
        "--mix-index",
        type=int,
        default=None,
        help="Optional dataset index to use as the secondary sample for mix-based augmentations.",
    )
    parser.add_argument(
        "--bootstrap-cutmix",
        type=int,
        default=32,
        help="How many random frames to scan when warming an empty InstanceCutMix bank.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def get_config_value(config: dict, key: str, default=None, section: str = None):
    if key in config:
        return config[key]
    if section is not None:
        section_cfg = config.get(section, {})
        if isinstance(section_cfg, dict) and key in section_cfg:
            return section_cfg[key]
    return default


def find_frame_index(dataset: SemanticKitti, seq_id: str, frame_id: str) -> int:
    seq_id = f"{int(seq_id):02d}"
    frame_id = f"{int(frame_id):06d}"
    for idx in range(len(dataset)):
        seq, frame = dataset.parsePathInfoByIndex(idx)
        if seq == seq_id and frame == frame_id:
            return idx
    raise ValueError(f"Frame not found for seq={seq_id} frame={frame_id}")


def build_mapped_color_lut(data_cfg: dict) -> np.ndarray:
    color_map_inv = data_cfg["color_map_inv"]
    max_key = max(int(k) for k in color_map_inv.keys())
    lut = np.zeros((max_key + 1, 3), dtype=np.uint8)
    for k, bgr in color_map_inv.items():
        lut[int(k)] = np.asarray(bgr[::-1], dtype=np.uint8)
    lut[0] = np.array([0, 0, 0], dtype=np.uint8)
    return lut


def save_image(array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, array.astype(np.uint8))


def create_projection(config: dict):
    sensor_cfg = config["sensor"]
    if sensor_cfg.get("scan_proj", False):
        return projection.ScanProjection(
            proj_w=sensor_cfg["proj_w"],
            proj_h=sensor_cfg["proj_h"],
        )
    return projection.RangeProjection(
        fov_up=sensor_cfg["fov_up"],
        fov_down=sensor_cfg["fov_down"],
        fov_left=sensor_cfg.get("fov_left", -180),
        fov_right=sensor_cfg.get("fov_right", 180),
        proj_w=sensor_cfg["proj_w"],
        proj_h=sensor_cfg["proj_h"],
    )


def project_semantic_range_image(pointcloud: np.ndarray, mapped_labels: np.ndarray, proj, color_lut: np.ndarray):
    _, _, proj_idx, proj_mask = proj.doProjection(pointcloud)
    out = np.zeros((proj_mask.shape[0], proj_mask.shape[1], 3), dtype=np.uint8)
    valid = proj_idx >= 0
    if np.any(valid):
        out[valid] = color_lut[np.clip(mapped_labels[proj_idx[valid]], 0, color_lut.shape[0] - 1)]
    return out, proj_mask.astype(bool)


def build_augmentor(config: dict):
    aug_cfg = config.get("augmentation", {})
    params = augmentor.AugmentParams()
    params.setFlipProb(
        p_flipx=aug_cfg.get("p_flipx", 0.0),
        p_flipy=aug_cfg.get("p_flipy", 0.0),
    )
    params.setTranslationParams(
        p_transx=aug_cfg.get("p_transx", 0.0),
        trans_xmin=aug_cfg.get("trans_xmin", 0.0),
        trans_xmax=aug_cfg.get("trans_xmax", 0.0),
        p_transy=aug_cfg.get("p_transy", 0.0),
        trans_ymin=aug_cfg.get("trans_ymin", 0.0),
        trans_ymax=aug_cfg.get("trans_ymax", 0.0),
        p_transz=aug_cfg.get("p_transz", 0.0),
        trans_zmin=aug_cfg.get("trans_zmin", 0.0),
        trans_zmax=aug_cfg.get("trans_zmax", 0.0),
    )
    params.setRotationParams(
        p_rot_roll=aug_cfg.get("p_rot_roll", 0.0),
        rot_rollmin=aug_cfg.get("rot_rollmin", 0.0),
        rot_rollmax=aug_cfg.get("rot_rollmax", 0.0),
        p_rot_pitch=aug_cfg.get("p_rot_pitch", 0.0),
        rot_pitchmin=aug_cfg.get("rot_pitchmin", 0.0),
        rot_pitchmax=aug_cfg.get("rot_pitchmax", 0.0),
        p_rot_yaw=aug_cfg.get("p_rot_yaw", 0.0),
        rot_yawmin=aug_cfg.get("rot_yawmin", 0.0),
        rot_yawmax=aug_cfg.get("rot_yawmax", 0.0),
    )
    if "p_scale" in aug_cfg:
        params.sefScaleParams(
            p_scale=aug_cfg.get("p_scale", 0.0),
            scale_min=aug_cfg.get("scale_min", 1.0),
            scale_max=aug_cfg.get("scale_max", 1.0),
        )
    return augmentor.Augmentor(params)


def build_crop_params(config: dict):
    model_cfg = config.get("model", {})
    image_size = tuple(model_cfg.get("image_size", config.get("image_size", [32, 384])))
    original_image_size = tuple(model_cfg.get("original_image_size", config.get("original_image_size", image_size)))
    train_full_image = bool(model_cfg.get("train_full_image", config.get("train_full_image", False)))
    p_hflip = float(config.get("augmentation", {}).get("p_hflip", 0.0))
    return image_size, original_image_size, train_full_image, p_hflip


def sample_crop_preview(pointcloud: np.ndarray, mapped_labels: np.ndarray, config: dict, proj, color_lut: np.ndarray):
    image_size, original_image_size, train_full_image, p_hflip = build_crop_params(config)
    crop_size = original_image_size if train_full_image else image_size
    _, proj_range, _, proj_mask = proj.doProjection(pointcloud)
    px = proj.cached_data["px"].copy()
    py = proj.cached_data["py"].copy()

    proj_mask_tensor = np.asarray(proj_mask, dtype=np.float32)
    proj_range_tensor = np.asarray(proj_range, dtype=np.float32)
    proj_tensor = np.concatenate(
        [
            proj_range_tensor[None, ...],
            np.zeros((5, proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32),
            proj_mask_tensor[None, ...],
        ],
        axis=0,
    )
    proj_label = np.zeros(proj_mask.shape, dtype=np.float32)
    valid = proj.cached_data["uproj_x_idx"].shape[0] == mapped_labels.shape[0]
    if not valid:
        return None
    _, _, proj_idx, _ = proj.doProjection(pointcloud)
    valid_proj = proj_idx >= 0
    proj_label[valid_proj] = mapped_labels[proj_idx[valid_proj]]
    proj_tensor[5] = proj_label

    proj_tensor_t = torch.from_numpy(proj_tensor)
    cropped, _, _, _, _ = crop_inputs(
        proj_tensor=proj_tensor_t,
        px=px,
        py=py,
        points_xyz=pointcloud[:, :3],
        labels=mapped_labels,
        crop_size=crop_size,
        center_crop=train_full_image,
        p_hflip=p_hflip,
    )
    cropped_np = cropped.numpy()
    crop_labels = cropped_np[5].astype(np.int32)
    crop_mask = cropped_np[6] > 0
    out = np.zeros((crop_labels.shape[0], crop_labels.shape[1], 3), dtype=np.uint8)
    out[crop_mask] = color_lut[np.clip(crop_labels[crop_mask], 0, color_lut.shape[0] - 1)]
    return out


def pick_mix_sample(dataset: SemanticKitti, frame_idx: int, rng: np.random.Generator, mix_index: int = None):
    if mix_index is not None:
        return dataset.loadDataByIndex(mix_index)
    if len(dataset) <= 1:
        return dataset.loadDataByIndex(frame_idx)
    candidates = [idx for idx in range(len(dataset)) if idx != frame_idx]
    chosen = int(rng.choice(candidates))
    return dataset.loadDataByIndex(chosen)


def maybe_map_labels(dataset: SemanticKitti, labels: np.ndarray, use_mapped: bool):
    return dataset.labelMapping(labels) if use_mapped else labels.copy()


def warmup_instance_cutmix_bank(instance_cutmix: InstanceCutMix, dataset: SemanticKitti, use_mapped: bool,
                                frame_idx: int, bootstrap_count: int, rng: np.random.Generator):
    if instance_cutmix.__loaded__ or bootstrap_count <= 0:
        return
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    seen = 0
    for idx in indices:
        if idx == frame_idx:
            continue
        pc, sem, inst = dataset.loadDataByIndex(idx)
        sem = maybe_map_labels(dataset, sem, use_mapped)
        instance_cutmix.cut(pc.copy(), sem.copy(), inst.copy())
        instance_cutmix.__loaded__ = all(len(instance_cutmix.bank[k]) > 0 for k in instance_cutmix.bank.keys())
        seen += 1
        if instance_cutmix.__loaded__ or seen >= bootstrap_count:
            break


def assemble_contact_sheet(images: list[np.ndarray], names: list[str], cell_pad: int = 8) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    cols = 2
    rows = int(math.ceil(len(images) / cols))
    text_band = 24
    sheet = np.full(
        (
            rows * (max_h + text_band + cell_pad) + cell_pad,
            cols * (max_w + cell_pad) + cell_pad,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
        ImageDraw = None

    if Image is not None:
        pil = Image.fromarray(sheet)
        draw = ImageDraw.Draw(pil)
        for idx, (img, name) in enumerate(zip(images, names)):
            row = idx // cols
            col = idx % cols
            y0 = cell_pad + row * (max_h + text_band + cell_pad)
            x0 = cell_pad + col * (max_w + cell_pad)
            pil.paste(Image.fromarray(img), (x0, y0 + text_band))
            draw.text((x0, y0 + 4), name, fill=(0, 0, 0))
        return np.asarray(pil)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = cell_pad + row * (max_h + text_band + cell_pad)
        x0 = cell_pad + col * (max_w + cell_pad)
        sheet[y0 + text_band:y0 + text_band + img.shape[0], x0:x0 + img.shape[1]] = img
    return sheet


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg_path = Path(args.data_config).resolve()
    data_cfg = yaml.safe_load(data_cfg_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()

    data_root_str = get_config_value(config, "data_root", section="data")
    if data_root_str is None:
        raise KeyError("Missing required config key: data.data_root")
    data_root = resolve_path(config_path.parent, data_root_str)

    seq_id = f"{int(args.seq):02d}"
    frame_id = f"{int(args.frame):06d}"
    use_mapped_labels = bool(config.get("adapted_augmentation", {}).get("use_mapped_labels", False))

    dataset = SemanticKitti(
        root=str(data_root),
        sequences=[int(seq_id)],
        config_path=str(data_cfg_path),
        has_label=bool(get_config_value(config, "has_label", True, section="data")),
    )
    frame_idx = find_frame_index(dataset, seq_id, frame_id)
    pointcloud, raw_labels, inst_labels = dataset.loadDataByIndex(frame_idx)
    mapped_labels = maybe_map_labels(dataset, raw_labels, use_mapped_labels)

    mix_pc, mix_sem_raw, mix_inst = pick_mix_sample(dataset, frame_idx, rng, args.mix_index)
    mix_sem = maybe_map_labels(dataset, mix_sem_raw, use_mapped_labels)

    color_lut = build_mapped_color_lut(data_cfg)
    proj = create_projection(config)

    previews: list[tuple[str, np.ndarray]] = []

    original_img, _ = project_semantic_range_image(pointcloud.copy(), mapped_labels.copy(), proj, color_lut)
    previews.append(("original", original_img))

    geometric_aug = build_augmentor(config)
    geometric_img, _ = project_semantic_range_image(
        geometric_aug.doAugmentation(pointcloud.copy()),
        mapped_labels.copy(),
        proj,
        color_lut,
    )
    previews.append(("geometric", geometric_img))

    adapted_cfg = config.get("adapted_augmentation", {})

    pointsample_cfg = adapted_cfg.get("pointsample", {})
    if pointsample_cfg.get("enable", False):
        sampler = augmentor.PointSampler(
            num_points=pointsample_cfg.get("num_points", 0),
            replace=pointsample_cfg.get("replace", False),
            inplace=True,
        )
        sampled_pc, sampled_labels, _ = sampler(pointcloud.copy(), mapped_labels.copy(), inst_labels.copy())
        sampled_img, _ = project_semantic_range_image(sampled_pc, sampled_labels, proj, color_lut)
        previews.append(("pointsample", sampled_img))

    polarmix_cfg = adapted_cfg.get("polarmix", {})
    if polarmix_cfg.get("enable", False):
        polarmix = PolarMix(
            classes=polarmix_cfg.get("classes", []),
            prob=polarmix_cfg.get("prob", 0.5),
        )
        mixed_pc, mixed_labels = polarmix(pointcloud.copy(), mapped_labels.copy(), mix_pc.copy(), mix_sem.copy())
        mixed_img, _ = project_semantic_range_image(mixed_pc, mixed_labels, proj, color_lut)
        previews.append(("polarmix", mixed_img))

    cutmix_cfg = adapted_cfg.get("instance_cutmix", {})
    if cutmix_cfg.get("enable", False):
        instance_cutmix = InstanceCutMix(
            rootdir=cutmix_cfg.get("instance_bank_root", "cache/instance_bank"),
            classes=cutmix_cfg.get("classes", []),
            num_to_add=tuple(cutmix_cfg.get("num_to_add", [0, 2])),
            min_points=cutmix_cfg.get("min_points", 20),
            prob=cutmix_cfg.get("prob", 1.0),
        )
        warmup_instance_cutmix_bank(
            instance_cutmix=instance_cutmix,
            dataset=dataset,
            use_mapped=use_mapped_labels,
            frame_idx=frame_idx,
            bootstrap_count=args.bootstrap_cutmix,
            rng=rng,
        )
        cut_pc, cut_labels = instance_cutmix(pointcloud.copy(), mapped_labels.copy(), inst_labels.copy())
        cut_img, _ = project_semantic_range_image(cut_pc, cut_labels, proj, color_lut)
        previews.append(("instance_cutmix", cut_img))

    clustermix_cfg = adapted_cfg.get("clustermix", {})
    if clustermix_cfg.get("enable", False):
        clustermix = ClusterMix(
            prob=clustermix_cfg.get("prob", 0.5),
            vertical_prob=clustermix_cfg.get("vertical_prob", 0.5),
            sector_width=clustermix_cfg.get("sector_width", np.pi / 2),
            height_ratio=clustermix_cfg.get("height_ratio", 0.5),
        )
        cluster_pc, cluster_labels = clustermix(pointcloud.copy(), mapped_labels.copy(), mix_pc.copy(), mix_sem.copy())
        cluster_img, _ = project_semantic_range_image(cluster_pc, cluster_labels, proj, color_lut)
        previews.append(("clustermix", cluster_img))

    instcopy_cfg = adapted_cfg.get("instance_copy", {})
    if instcopy_cfg.get("enable", False):
        instance_copy = InstanceCopy(
            prob=instcopy_cfg.get("prob", 0.5),
            classes=instcopy_cfg.get("classes", []),
            max_instances_per_class=instcopy_cfg.get("max_instances_per_class", 1),
        )
        copied_pc, copied_labels = instance_copy(
            pointcloud.copy(),
            mapped_labels.copy(),
            mix_pc.copy(),
            mix_sem.copy(),
            mix_inst.copy(),
        )
        copied_img, _ = project_semantic_range_image(copied_pc, copied_labels, proj, color_lut)
        previews.append(("instance_copy", copied_img))

    crop_img = sample_crop_preview(pointcloud.copy(), mapped_labels.copy(), config, proj, color_lut)
    if crop_img is not None:
        previews.append(("random_crop", crop_img))

    run_dir = output_dir / f"seq{seq_id}_frame{frame_id}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    names = []
    images = []
    for name, img in previews:
        names.append(name)
        images.append(img)
        save_image(img, run_dir / f"{name}.png")

    contact_sheet = assemble_contact_sheet(images, names)
    save_image(contact_sheet, run_dir / "summary.png")

    print(f"Saved {len(previews)} previews to {run_dir}")
    print("Methods:")
    for name in names:
        print(f"  - {name}")
    print(f"Summary sheet: {run_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
