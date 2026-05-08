import argparse
import json
import os
from pathlib import Path

import torch

from src.backbones import get_model
from src.detection_mvtecfs import run_mvtecfs_inference_jsonl
from src.mvtec_fs import MVTecFSInstanceDataset, sample_support_by_class


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_mvtecfs_anomalydino")
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mv_method", type=str, default="mso", choices=["none", "mso"])
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--heatmap_class", type=str, default="pred", choices=["pred", "gt"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    model = get_model(
        args.model_name,
        args.device,
        smaller_edge_size=args.image_size,
        square_resize=True,
    )

    output_root = Path(args.output_dir) / args.model_name
    output_root.mkdir(parents=True, exist_ok=True)

    metrics = []

    for object_name in args.objects:
        train_ds = MVTecFSInstanceDataset(args.data_root, object_name, "train")
        valid_ds = MVTecFSInstanceDataset(args.data_root, object_name, "valid")

        for shot in args.shots:
            for seed in range(args.num_seeds):
                support_items = sample_support_by_class(train_ds, shot, seed)
                query_items = valid_ds.items

                output_jsonl = output_root / f"{object_name}_{shot}shot_seed{seed}.jsonl"

                result = run_mvtecfs_inference_jsonl(
                    model,
                    support_items,
                    query_items,
                    str(output_jsonl),
                    model_name=args.model_name,
                    image_size=args.image_size,
                    mv_method=args.mv_method,
                    k_neighbors=args.k_neighbors,
                    heatmap_class=args.heatmap_class,
                    shot=shot,
                    seed=seed,
                )
                result["object_name"] = object_name
                result["shot"] = shot
                result["seed"] = seed
                metrics.append(result)
                print(result)

    metrics_path = output_root / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
