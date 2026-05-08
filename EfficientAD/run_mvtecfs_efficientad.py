from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import get_autoencoder, get_pdn_medium, get_pdn_small


out_channels = 384


DEFAULT_OBJECTS = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class MVTecFSRecord:
    image_path: Path
    rel_image_path: str
    output_image_path: str
    label: str
    split: str
    object_name: str
    defect_name: str
    image_width: int
    image_height: int


class MVTecFSDataset(Dataset):
    def __init__(self, records: list[MVTecFSRecord], transform: Any) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return {"image": self.transform(image), "record": record}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EfficientAD MVTec-FS heatmaps as work-1 JSONL.")
    parser.add_argument("--data_root", required=True, help="MVTec-FS dataset root.")
    parser.add_argument("--output_dir", default="results_mvtecfs_efficientad")
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    parser.add_argument("--split", default="valid", help="MVTec-FS CONFIG split to run, usually valid.")
    parser.add_argument("--normalization-split", default="train")
    parser.add_argument("--model-size", default="small", choices=["small", "medium"])
    parser.add_argument("--teacher-weights", default=None)
    parser.add_argument(
        "--student-template",
        default="output/1/trainings/mvtec_ad/{object}/student_final.pth",
        help="Per-object student path template; supports {object}, {model_size}.",
    )
    parser.add_argument(
        "--autoencoder-template",
        default="output/1/trainings/mvtec_ad/{object}/autoencoder_final.pth",
        help="Per-object autoencoder path template; supports {object}, {model_size}.",
    )
    parser.add_argument(
        "--teacher-template",
        default=None,
        help="Optional per-object teacher path template. Defaults to --teacher-weights.",
    )
    parser.add_argument("--missing-model-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-resolution", choices=["model", "image"], default="model")
    parser.add_argument("--normalize", choices=["efficientad-quantile", "per-image-minmax", "none"], default="efficientad-quantile")
    parser.add_argument("--localizer", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    localizer = args.localizer or f"efficientad_{args.model_size}"
    transform = build_transform(args.image_size)

    output_root = Path(args.output_dir) / localizer
    output_root.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    for object_name in args.objects:
        paths = resolve_model_paths(args, object_name)
        missing = [name for name, path in paths.items() if path is not None and not path.exists()]
        if missing:
            message = ", ".join(f"{name}={paths[name]}" for name in missing)
            if args.missing_model_policy == "skip":
                print(f"SKIP {object_name}: missing model files: {message}", flush=True)
                continue
            raise FileNotFoundError(f"Missing EfficientAD model files for {object_name}: {message}")

        teacher, student, autoencoder = load_models(args, paths, device)
        norm_records = load_mvtecfs_records(Path(args.data_root), object_name, args.normalization_split)
        query_records = load_mvtecfs_records(Path(args.data_root), object_name, args.split)
        if not query_records:
            print(f"SKIP {object_name}: no records for split={args.split}", flush=True)
            continue

        normalization_loader = DataLoader(
            MVTecFSDataset(norm_records, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        query_loader = DataLoader(
            MVTecFSDataset(query_records, transform),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )

        teacher_mean, teacher_std = teacher_normalization(teacher, normalization_loader, device, object_name)
        q_st_start = q_st_end = q_ae_start = q_ae_end = None
        if args.normalize == "efficientad-quantile":
            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                normalization_loader,
                teacher,
                student,
                autoencoder,
                teacher_mean,
                teacher_std,
                device,
                object_name,
            )

        output_jsonl = output_root / f"{object_name}_{args.split}.jsonl"
        written = run_inference_jsonl(
            query_loader,
            teacher,
            student,
            autoencoder,
            teacher_mean,
            teacher_std,
            q_st_start,
            q_st_end,
            q_ae_start,
            q_ae_end,
            output_jsonl,
            args,
            localizer,
            device,
        )
        result = {
            "object_name": object_name,
            "split": args.split,
            "records": written,
            "jsonl": str(output_jsonl),
            "student": str(paths["student"]),
            "autoencoder": str(paths["autoencoder"]),
            "teacher": str(paths["teacher"]),
        }
        metrics.append(result)
        print(result, flush=True)

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


def load_models(args: argparse.Namespace, paths: dict[str, Path | None], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    teacher = load_or_build_model(
        paths["teacher"],
        build_teacher(args.model_size),
        device,
        description="teacher",
    )
    student = load_or_build_model(
        paths["student"],
        build_student(args.model_size),
        device,
        description="student",
    )
    autoencoder = load_or_build_model(
        paths["autoencoder"],
        get_autoencoder(out_channels),
        device,
        description="autoencoder",
    )
    teacher.eval()
    student.eval()
    autoencoder.eval()
    return teacher, student, autoencoder


def build_teacher(model_size: str) -> torch.nn.Module:
    if model_size == "small":
        return get_pdn_small(out_channels)
    if model_size == "medium":
        return get_pdn_medium(out_channels)
    raise ValueError(f"Unsupported model size: {model_size}")


def build_student(model_size: str) -> torch.nn.Module:
    if model_size == "small":
        return get_pdn_small(2 * out_channels)
    if model_size == "medium":
        return get_pdn_medium(2 * out_channels)
    raise ValueError(f"Unsupported model size: {model_size}")


def load_or_build_model(path: Path | None, fallback: torch.nn.Module, device: torch.device, description: str) -> torch.nn.Module:
    if path is None:
        raise ValueError(f"Missing path for {description}")
    loaded = torch_load(path)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported {description} checkpoint format: {path}")
        model = fallback
        model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    return model.to(device)


@torch.no_grad()
def predict(
    image: torch.Tensor,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    autoencoder: torch.nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    q_st_start: torch.Tensor | None = None,
    q_st_end: torch.Tensor | None = None,
    q_ae_start: torch.Tensor | None = None,
    q_ae_end: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:]) ** 2, dim=1, keepdim=True)
    if q_st_start is not None and q_st_end is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None and q_ae_end is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in state_dict.items():
        output[key.replace("module.", "", 1) if key.startswith("module.") else key] = value
    return output


@torch.no_grad()
def teacher_normalization(
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_outputs = []
    for batch in tqdm(loader, desc=f"{object_name}: teacher mean"):
        images = batch["image"].to(device, non_blocking=True)
        teacher_output = teacher(images)
        mean_outputs.append(torch.mean(teacher_output, dim=[0, 2, 3]))
    if not mean_outputs:
        raise ValueError(f"No normalization images found for {object_name}")
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)[None, :, None, None]

    mean_distances = []
    for batch in tqdm(loader, desc=f"{object_name}: teacher std"):
        images = batch["image"].to(device, non_blocking=True)
        teacher_output = teacher(images)
        distance = (teacher_output - channel_mean) ** 2
        mean_distances.append(torch.mean(distance, dim=[0, 2, 3]))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)[None, :, None, None]
    channel_std = torch.sqrt(channel_var).clamp_min(1e-12)
    return channel_mean, channel_std


@torch.no_grad()
def map_normalization(
    loader: DataLoader,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    autoencoder: torch.nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    device: torch.device,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maps_st = []
    maps_ae = []
    for batch in tqdm(loader, desc=f"{object_name}: map normalization"):
        images = batch["image"].to(device, non_blocking=True)
        _, map_st, map_ae = predict(
            image=images,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st_cat = torch.cat(maps_st)
    maps_ae_cat = torch.cat(maps_ae)
    return (
        torch.quantile(maps_st_cat, q=0.9),
        torch.quantile(maps_st_cat, q=0.995),
        torch.quantile(maps_ae_cat, q=0.9),
        torch.quantile(maps_ae_cat, q=0.995),
    )


@torch.no_grad()
def run_inference_jsonl(
    loader: DataLoader,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    autoencoder: torch.nn.Module,
    teacher_mean: torch.Tensor,
    teacher_std: torch.Tensor,
    q_st_start: torch.Tensor | None,
    q_st_end: torch.Tensor | None,
    q_ae_start: torch.Tensor | None,
    q_ae_end: torch.Tensor | None,
    output_jsonl: Path,
    args: argparse.Namespace,
    localizer: str,
    device: torch.device,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for batch in tqdm(loader, desc=f"inference -> {output_jsonl.name}"):
            images = batch["image"].to(device, non_blocking=True)
            record = batch["record"][0]
            map_combined, _, _ = predict(
                image=images,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=q_st_start,
                q_st_end=q_st_end,
                q_ae_start=q_ae_start,
                q_ae_end=q_ae_end,
            )
            map_combined = F.pad(map_combined, (4, 4, 4, 4))
            if args.heatmap_resolution == "image":
                map_combined = F.interpolate(
                    map_combined,
                    (record.image_height, record.image_width),
                    mode="bilinear",
                    align_corners=False,
                )
            heatmap = map_combined[0, 0].detach().cpu().numpy().astype(np.float32)
            if args.normalize == "per-image-minmax":
                heatmap = minmax_normalize(heatmap)
            write_heatmap_record(handle, record, heatmap, args, localizer)
            written += 1
    return written


def write_heatmap_record(
    handle: Any,
    record: MVTecFSRecord,
    heatmap: np.ndarray,
    args: argparse.Namespace,
    localizer: str,
) -> None:
    payload = {
        "image_path": record.output_image_path,
        "label": record.label,
        "split": record.split,
        "object_name": record.object_name,
        "defect_name": record.defect_name,
        "image_width": record.image_width,
        "image_height": record.image_height,
        "heatmap_width": int(heatmap.shape[1]),
        "heatmap_height": int(heatmap.shape[0]),
        "heatmap": np.round(heatmap, 6).tolist(),
        "localizer": localizer,
        "model": f"efficientad_{args.model_size}",
        "image_size": args.image_size,
        "score_normalization": args.normalize,
        "heatmap_resolution": args.heatmap_resolution,
    }
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_mvtecfs_records(root: Path, object_name: str, split: str) -> list[MVTecFSRecord]:
    csv_path = root / "CONFIG" / f"{object_name}_config1" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MVTec-FS CSV: {csv_path}")

    records: list[MVTecFSRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel_path = first_nonempty(row, ["img_rel_path", "image_path", "path", "image", "img"])
            if not rel_path:
                raise ValueError(f"Missing image path in {csv_path}")
            image_path = resolve_image_path(root, object_name, rel_path)
            width, height = parse_image_size(row, image_path)
            label = first_nonempty(row, ["label", "defect_name", "class", "category"]) or Path(rel_path).parent.name
            rel_path_norm = rel_path.replace("\\", "/").lstrip("/")
            records.append(
                MVTecFSRecord(
                    image_path=image_path,
                    rel_image_path=rel_path_norm,
                    output_image_path=f"image/{object_name}/{rel_path_norm}",
                    label=label,
                    split=split,
                    object_name=object_name,
                    defect_name=label,
                    image_width=width,
                    image_height=height,
                )
            )
    return records


def resolve_image_path(root: Path, object_name: str, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    candidates = [root / "image" / object_name / rel_path, root / rel_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for {rel_path}; tried: {candidates}")


def parse_image_size(row: dict[str, str], image_path: Path) -> tuple[int, int]:
    width = parse_int(first_nonempty(row, ["imageWidth", "image_width", "width", "w"]))
    height = parse_int(first_nonempty(row, ["imageHeight", "image_height", "height", "h"]))
    if width is not None and height is not None:
        return width, height
    with Image.open(image_path) as image:
        return image.size


def first_nonempty(row: dict[str, str], fields: list[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "record": [item["record"] for item in items],
    }


def resolve_model_paths(args: argparse.Namespace, object_name: str) -> dict[str, Path | None]:
    model_size = args.model_size
    teacher_default = Path(args.teacher_weights or f"models/teacher_{model_size}.pth")
    teacher = render_template(args.teacher_template, object_name, model_size) if args.teacher_template else teacher_default
    return {
        "teacher": teacher,
        "student": render_template(args.student_template, object_name, model_size),
        "autoencoder": render_template(args.autoencoder_template, object_name, model_size),
    }


def render_template(template: str, object_name: str, model_size: str) -> Path:
    return Path(template.format(object=object_name, class_name=object_name, model_size=model_size))


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        device_text = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_text)
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
    return device


if __name__ == "__main__":
    main()
