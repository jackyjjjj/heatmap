import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.mvtec_fs import MVTecFSItem
from src.mvtecfs_views import make_mso_views, patch_mask_from_instance_mask, polygon_to_mask
from src.post_eval import mean_top1p

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class ClassMemoryBank:
    def __init__(self, dim: int, device: str = "cpu", use_faiss: bool = True) -> None:
        self.dim = dim
        self.device = device
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._features_by_label: Dict[str, List[np.ndarray]] = {}
        self._faiss_index: Dict[str, "faiss.IndexFlatIP"] = {}
        self._torch_features: Dict[str, torch.Tensor] = {}
        self.labels: List[str] = []

    def add(self, label: str, features: np.ndarray) -> None:
        if features is None or len(features) == 0:
            return
        self._features_by_label.setdefault(label, []).append(features.astype(np.float32, copy=False))

    def build(self) -> None:
        self.labels = sorted(self._features_by_label.keys())
        for label in self.labels:
            feats = np.concatenate(self._features_by_label[label], axis=0).astype(np.float32, copy=False)
            if feats.shape[1] != self.dim:
                raise ValueError(f"Feature dim mismatch for label '{label}': {feats.shape[1]} != {self.dim}")
            if self.use_faiss:
                faiss.normalize_L2(feats)
                index = faiss.IndexFlatIP(feats.shape[1])
                index.add(feats)
                self._faiss_index[label] = index
            else:
                tensor = torch.from_numpy(feats)
                tensor = F.normalize(tensor, dim=1)
                if self.device:
                    tensor = tensor.to(self.device)
                self._torch_features[label] = tensor

    def search_label(self, label: str, query_features: np.ndarray, k: int = 1) -> np.ndarray:
        if label not in self.labels:
            raise ValueError(f"Label '{label}' not found in memory bank.")
        if self.use_faiss:
            q = query_features.astype(np.float32, copy=False)
            faiss.normalize_L2(q)
            scores, _ = self._faiss_index[label].search(q, k)
            if k > 1:
                scores = scores.mean(axis=1)
            return scores.squeeze()

        q = torch.from_numpy(query_features.astype(np.float32, copy=False))
        q = F.normalize(q, dim=1)
        if self.device:
            q = q.to(self.device)
        mem = self._torch_features[label]
        sims = q @ mem.T
        if k > 1:
            scores = sims.topk(k, dim=1).values.mean(dim=1)
        else:
            scores = sims.max(dim=1).values
        return scores.detach().cpu().numpy()


def extract_support_defect_patches(
    model,
    item: MVTecFSItem,
    image_size: int,
    mv_method: str = "mso",
    mask_threshold: float = 0.10,
) -> np.ndarray:
    image = Image.open(item.image_path).convert("RGB")
    mask, bbox = _instance_mask_and_bbox(item)

    if mv_method == "mso":
        views = make_mso_views(image, mask, bbox)
    else:
        views = [(image, mask, bbox)]

    features_all = []
    for view_img, view_mask, _ in views:
        image_tensor, grid_size = model.prepare_image(view_img)
        features = model.extract_features(image_tensor)
        patch_mask = patch_mask_from_instance_mask(view_mask, grid_size, threshold=mask_threshold)
        if patch_mask.sum() == 0:
            continue
        features_all.append(features[patch_mask])

    if not features_all:
        dim = _infer_embed_dim(model)
        return np.zeros((0, dim), dtype=np.float32)

    return np.concatenate(features_all, axis=0)


def build_mvtecfs_memory_bank(
    model,
    support_items: Iterable[MVTecFSItem],
    image_size: int,
    mv_method: str = "mso",
    k_neighbors: int = 1,
    use_faiss: bool = True,
) -> ClassMemoryBank:
    dim = _infer_embed_dim(model)
    bank = ClassMemoryBank(dim=dim, device=model.device, use_faiss=use_faiss)

    expected_labels = set()
    for item in support_items:
        expected_labels.add(item.label)
        features = extract_support_defect_patches(
            model,
            item,
            image_size=image_size,
            mv_method=mv_method,
        )
        bank.add(item.label, features)

    bank.build()
    missing = expected_labels.difference(bank.labels)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"No features extracted for labels: {missing_str}")
    return bank


def infer_one_heatmap(
    model,
    bank: ClassMemoryBank,
    item: MVTecFSItem,
    k_neighbors: int = 1,
    heatmap_class: str = "pred",
) -> Tuple[np.ndarray, str, Dict[str, float], Tuple[int, int]]:
    image = Image.open(item.image_path).convert("RGB")
    image_tensor, grid_size = model.prepare_image(image)
    features = model.extract_features(image_tensor)

    class_scores: Dict[str, float] = {}
    score_maps: Dict[str, np.ndarray] = {}

    for label in bank.labels:
        patch_scores = bank.search_label(label, features, k=k_neighbors)
        score_map = patch_scores.reshape(grid_size)
        score_maps[label] = score_map
        class_scores[label] = float(mean_top1p(score_map))

    pred_label = max(class_scores, key=class_scores.get)
    target_label = pred_label if heatmap_class == "pred" else item.label
    if target_label not in score_maps:
        target_label = pred_label

    heatmap = score_maps[target_label]
    heatmap = _minmax_normalize(heatmap)

    return heatmap, pred_label, class_scores, grid_size


def write_heatmap_record(
    fp,
    item: MVTecFSItem,
    heatmap: np.ndarray,
    pred_label: str,
    class_scores: Dict[str, float],
    model_name: str,
    image_size: int,
    localizer: str = "dinov2_patch-contrast",
    shot: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    rel_path = item.rel_image_path.replace("\\", "/")
    rel_path = rel_path.lstrip("/")
    image_path = f"image/{item.object_name}/{rel_path}"
    heatmap_rounded = np.round(heatmap.astype(np.float32), 6).tolist()
    record = {
        "image_path": image_path,
        "label": item.label,
        "split": item.split,
        "object_name": item.object_name,
        "defect_name": item.defect_name,
        "image_width": item.image_width,
        "image_height": item.image_height,
        "heatmap_width": heatmap.shape[1],
        "heatmap_height": heatmap.shape[0],
        "heatmap": heatmap_rounded,
        "localizer": localizer,
        "model": model_name,
        "image_size": image_size,
        "score_normalization": "per_image_minmax",
        "pred_label": pred_label,
        "class_scores": {k: float(round(v, 6)) for k, v in class_scores.items()},
    }
    if shot is not None:
        record["shot"] = shot
    if seed is not None:
        record["seed"] = seed

    fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_mvtecfs_inference_jsonl(
    model,
    support_items: Iterable[MVTecFSItem],
    query_items: Iterable[MVTecFSItem],
    output_jsonl: str,
    model_name: str,
    image_size: int,
    mv_method: str = "mso",
    k_neighbors: int = 1,
    heatmap_class: str = "pred",
    shot: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    bank = build_mvtecfs_memory_bank(
        model,
        support_items,
        image_size=image_size,
        mv_method=mv_method,
        k_neighbors=k_neighbors,
    )

    correct = 0
    total = 0

    with open(output_jsonl, "w", encoding="utf-8") as fp:
        for item in query_items:
            heatmap, pred_label, class_scores, _ = infer_one_heatmap(
                model,
                bank,
                item,
                k_neighbors=k_neighbors,
                heatmap_class=heatmap_class,
            )
            write_heatmap_record(
                fp,
                item,
                heatmap,
                pred_label,
                class_scores,
                model_name=model_name,
                image_size=image_size,
                shot=shot,
                seed=seed,
            )

            total += 1
            if pred_label == item.label:
                correct += 1

    accuracy = float(correct) / float(total) if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "jsonl": output_jsonl,
    }


def _instance_mask_and_bbox(item: MVTecFSItem) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if item.points:
        mask = polygon_to_mask(item.points, item.image_width, item.image_height)
    elif item.bbox:
        mask = np.zeros((item.image_height, item.image_width), dtype=np.uint8)
        x1, y1, x2, y2 = _clamp_bbox(item.bbox, item.image_width, item.image_height)
        mask[y1:y2, x1:x2] = 255
    else:
        mask = np.ones((item.image_height, item.image_width), dtype=np.uint8) * 255

    if item.bbox:
        bbox = item.bbox
    else:
        bbox = _bbox_from_mask(mask, item.image_width, item.image_height)

    return mask, bbox


def _bbox_from_mask(mask: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


def _infer_embed_dim(model) -> int:
    dim = getattr(model.model, "embed_dim", None)
    if dim is None:
        dim = getattr(model.model, "hidden_dim", None)
    if dim is None:
        raise ValueError("Unable to infer embedding dimension from model.")
    return int(dim)
