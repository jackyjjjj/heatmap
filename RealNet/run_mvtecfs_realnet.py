from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from PIL import Image
from torchvision import transforms

from models.model_helper import ModelHelper
from utils.misc_helper import set_seed


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


class MVTecFSDataset(torch.utils.data.Dataset):
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
    parser = argparse.ArgumentParser(description="Export RealNet MVTec-FS heatmaps as work-1 JSONL.")
    parser.add_argument("--data_root", required=True, help="MVTec-FS dataset root.")
    parser.add_argument("--output_dir", default="results_mvtecfs_realnet")
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    parser.add_argument("--split", default="valid", help="MVTec-FS CONFIG split to run, usually valid.")
    parser.add_argument("--config", default="experiments/MVTec-AD/realnet.yaml")
    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help=(
            "Checkpoint path template. Supports {object}, {class_name}, {config_dir}, {checkpoint_root}. "
            "Default: {config_dir}/realnet_checkpoints/{object}/ckpt_best.pth.tar"
        ),
    )
    parser.add_argument("--checkpoint-root", default=None, help="Optional root used by --checkpoint-template.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--heatmap-resolution", choices=["model", "image"], default="model")
    parser.add_argument("--normalize", choices=["per-image-minmax", "none"], default="per-image-minmax")
    parser.add_argument("--missing-checkpoint-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--localizer", default="realnet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    config.exp_path = str(config_path.parent)
    configure_realnet(config)
    set_seed(int(config.get("random_seed", 100)))

    transform = build_transform(config.dataset.input_size, config.dataset.pixel_mean, config.dataset.pixel_std)
    output_root = Path(args.output_dir) / args.localizer
    output_root.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    for object_name in args.objects:
        checkpoint = resolve_checkpoint_path(args, config_path, object_name)
        if not checkpoint.exists():
            message = f"Missing RealNet checkpoint for {object_name}: {checkpoint}"
            if args.missing_checkpoint_policy == "skip":
                print(f"SKIP: {message}", flush=True)
                continue
            raise FileNotFoundError(message)

        records = load_mvtecfs_records(Path(args.data_root), object_name, args.split)
        if not records:
            print(f"SKIP: no records for {object_name} split={args.split}", flush=True)
            continue

        model = build_model(config, checkpoint, device)
        dataset = MVTecFSDataset(records, transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )

        output_jsonl = output_root / f"{object_name}_{args.split}.jsonl"
        written = run_inference_jsonl(
            model=model,
            loader=loader,
            output_jsonl=output_jsonl,
            device=device,
            args=args,
        )
        result = {
            "object_name": object_name,
            "split": args.split,
            "records": written,
            "jsonl": str(output_jsonl),
            "checkpoint": str(checkpoint),
        }
        metrics.append(result)
        print(result, flush=True)

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


def configure_realnet(config: EasyDict) -> None:
    layers: list[str] = []
    for block in config.structure:
        layers.extend([layer.idx for layer in block.layers])
    layers = sorted(set(layers))

    config.net[0].kwargs["outlayers"] = layers
    config.net[1].kwargs = config.net[1].get("kwargs", {})
    config.net[1].kwargs["structure"] = config.structure


def build_model(config: EasyDict, checkpoint: Path, device: torch.device) -> ModelHelper:
    model = ModelHelper(config.net)
    if device.type == "cuda":
        model.cuda()
    else:
        model.cpu()

    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in state_dict.items():
        output[key.replace("module.", "", 1) if key.startswith("module.") else key] = value
    return output


def run_inference_jsonl(
    model: ModelHelper,
    loader: torch.utils.data.DataLoader,
    output_jsonl: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                outputs = model({"image": images}, train=False)
                maps = outputs["anomaly_score"].detach().float()
                if maps.ndim == 4:
                    maps = maps[:, 0]

                for index, record in enumerate(batch["record"]):
                    heatmap_tensor = maps[index]
                    if args.heatmap_resolution == "image":
                        heatmap_tensor = F.interpolate(
                            heatmap_tensor[None, None],
                            size=(record.image_height, record.image_width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                    heatmap = heatmap_tensor.cpu().numpy().astype(np.float32)
                    if args.normalize == "per-image-minmax":
                        heatmap = minmax_normalize(heatmap)
                    write_heatmap_record(handle, record, heatmap, args)
                    written += 1
                print(f"extracted {written}/{len(loader.dataset)} -> {output_jsonl}", flush=True)
    return written


def write_heatmap_record(handle: Any, record: MVTecFSRecord, heatmap: np.ndarray, args: argparse.Namespace) -> None:
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
        "localizer": args.localizer,
        "model": "realnet",
        "score_normalization": args.normalize,
        "heatmap_resolution": args.heatmap_resolution,
    }
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_transform(input_size: list[int], mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(tuple(input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "record": [item["record"] for item in items],
    }


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


def resolve_checkpoint_path(args: argparse.Namespace, config_path: Path, object_name: str) -> Path:
    config_dir = config_path.parent
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else config_dir / "realnet_checkpoints"
    template = args.checkpoint_template or "{checkpoint_root}/{object}/ckpt_best.pth.tar"
    rendered = template.format(
        object=object_name,
        class_name=object_name,
        config_dir=str(config_dir),
        checkpoint_root=str(checkpoint_root),
    )
    return Path(rendered)


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
