import csv
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image


IMAGE_PATH_FIELDS = ["img_rel_path", "imagePath", "path", "image_path", "image", "img_path", "img"]
LABEL_FIELDS = ["label", "class", "defect_name", "y", "category", "defect"]
POINTS_FIELDS = ["points", "polygon", "segmentation", "poly"]
JSON_FIELDS = ["json_path", "labelme_json", "json", "anno_path", "annotation", "annotation_path"]
WIDTH_FIELDS = ["imageWidth", "image_width", "width", "w"]
HEIGHT_FIELDS = ["imageHeight", "image_height", "height", "h"]


@dataclass
class MVTecFSItem:
    image_path: str
    rel_image_path: str
    label: str
    split: str
    object_name: str
    defect_name: str
    bbox: Optional[Tuple[int, int, int, int]]
    points: Optional[List[List[float]]]
    image_width: int
    image_height: int
    instance_id: Optional[str] = None


class MVTecFSInstanceDataset:
    def __init__(self, root: str, object_name: str, split: str) -> None:
        self.root = root
        self.object_name = object_name
        self.split = split

        config_dir = os.path.join(root, "CONFIG", f"{object_name}_config1")
        csv_path = os.path.join(config_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        self.items = self._load_csv(csv_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> MVTecFSItem:
        return self.items[idx]

    def _load_csv(self, csv_path: str) -> List[MVTecFSItem]:
        items: List[MVTecFSItem] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_items = self._items_from_row(row)
                items.extend(row_items)
        return items

    def _items_from_row(self, row: dict) -> List[MVTecFSItem]:
        rel_path = _get_first_nonempty(row, IMAGE_PATH_FIELDS)
        label = _get_first_nonempty(row, LABEL_FIELDS)
        points = _parse_points(_get_first_nonempty(row, POINTS_FIELDS))
        bbox = _parse_bbox(row)
        json_path = _get_first_nonempty(row, JSON_FIELDS)
        instance_id = row.get("id") or row.get("img_id") or row.get("instance_id")

        if not label and rel_path:
            label = os.path.basename(os.path.dirname(rel_path))

        if json_path and not os.path.isabs(json_path):
            json_path = os.path.join(self.root, json_path)

        if (points is None) and (bbox is None) and json_path and os.path.exists(json_path):
            shapes = _load_labelme_shapes(json_path)
            if label:
                shapes = [s for s in shapes if s["label"] == label]
            if shapes:
                return [
                    _make_item_from_shape(
                        self.root,
                        self.object_name,
                        self.split,
                        row,
                        rel_path,
                        shape,
                        instance_id,
                    )
                    for shape in shapes
                ]

        return [
            _make_item(
                self.root,
                self.object_name,
                self.split,
                row,
                rel_path,
                label,
                bbox,
                points,
                instance_id,
            )
        ]


def _get_first_nonempty(row: dict, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_points(points_str: Optional[str]) -> Optional[List[List[float]]]:
    if not points_str:
        return None
    try:
        points = json.loads(points_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(points, list):
        return None
    return points


def _parse_bbox(row: dict) -> Optional[Tuple[int, int, int, int]]:
    if not all(k in row for k in ("x1", "y1", "x2", "y2")):
        return None
    try:
        x1 = int(float(row["x1"]))
        y1 = int(float(row["y1"]))
        x2 = int(float(row["x2"]))
        y2 = int(float(row["y2"]))
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _load_labelme_shapes(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    output = []
    for shape in shapes:
        label = shape.get("label")
        points = shape.get("points")
        if not label or not points:
            continue
        output.append({"label": label, "points": points})
    return output


def _resolve_image_path(root: str, object_name: str, rel_path: Optional[str]) -> str:
    if not rel_path:
        raise ValueError("Missing image path in CSV row.")
    if os.path.isabs(rel_path):
        return rel_path

    candidates = [
        os.path.join(root, "image", object_name, rel_path),
        os.path.join(root, rel_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _parse_size(row: dict, image_path: str) -> Tuple[int, int]:
    width = _parse_int_from_fields(row, WIDTH_FIELDS)
    height = _parse_int_from_fields(row, HEIGHT_FIELDS)
    if width is not None and height is not None:
        return width, height
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def _parse_int_from_fields(row: dict, fields: List[str]) -> Optional[int]:
    for key in fields:
        if key in row and row[key] not in (None, ""):
            try:
                return int(float(row[key]))
            except (TypeError, ValueError):
                return None
    return None


def _bbox_from_points(points: List[List[float]]) -> Optional[Tuple[int, int, int, int]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _make_item(
    root: str,
    object_name: str,
    split: str,
    row: dict,
    rel_path: Optional[str],
    label: Optional[str],
    bbox: Optional[Tuple[int, int, int, int]],
    points: Optional[List[List[float]]],
    instance_id: Optional[str],
) -> MVTecFSItem:
    image_path = _resolve_image_path(root, object_name, rel_path)
    width, height = _parse_size(row, image_path)
    if points and not bbox:
        bbox = _bbox_from_points(points)
    if not label:
        label = "unknown"
    return MVTecFSItem(
        image_path=image_path,
        rel_image_path=rel_path or os.path.basename(image_path),
        label=label,
        split=split,
        object_name=object_name,
        defect_name=label,
        bbox=bbox,
        points=points,
        image_width=width,
        image_height=height,
        instance_id=instance_id,
    )


def _make_item_from_shape(
    root: str,
    object_name: str,
    split: str,
    row: dict,
    rel_path: Optional[str],
    shape: dict,
    instance_id: Optional[str],
) -> MVTecFSItem:
    label = shape["label"]
    points = shape["points"]
    bbox = _bbox_from_points(points)
    return _make_item(
        root,
        object_name,
        split,
        row,
        rel_path,
        label,
        bbox,
        points,
        instance_id,
    )


def sample_support_by_class(train_dataset: MVTecFSInstanceDataset, k_shot: int, seed: int) -> List[MVTecFSItem]:
    rng = random.Random(seed)
    by_label: dict = {}
    for item in train_dataset.items:
        by_label.setdefault(item.label, []).append(item)

    support_items: List[MVTecFSItem] = []
    for label, items in by_label.items():
        items_sorted = sorted(items, key=lambda i: i.rel_image_path)
        if len(items_sorted) < k_shot:
            raise ValueError(f"Not enough samples for label '{label}': {len(items_sorted)} < {k_shot}")
        support_items.extend(rng.sample(items_sorted, k_shot))

    return support_items
