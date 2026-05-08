from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(points: Iterable, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    poly = [(float(x), float(y)) for x, y in points]
    draw.polygon(poly, outline=255, fill=255)
    return np.array(mask, dtype=np.uint8)


def crop_with_padding(img, box: Tuple[float, float, float, float], fill=0):
    x1, y1, x2, y2 = box
    x1_i = int(np.floor(x1))
    y1_i = int(np.floor(y1))
    x2_i = int(np.ceil(x2))
    y2_i = int(np.ceil(y2))

    if isinstance(img, Image.Image):
        img_np = np.array(img)
        is_pil = True
        mode = img.mode
    else:
        img_np = np.array(img)
        is_pil = False
        mode = None

    h, w = img_np.shape[:2]
    out_w = max(1, x2_i - x1_i)
    out_h = max(1, y2_i - y1_i)

    if img_np.ndim == 2:
        out = np.full((out_h, out_w), fill, dtype=img_np.dtype)
    else:
        fill_value = fill if isinstance(fill, (list, tuple, np.ndarray)) else [fill] * img_np.shape[2]
        out = np.zeros((out_h, out_w, img_np.shape[2]), dtype=img_np.dtype)
        out[:] = np.array(fill_value, dtype=img_np.dtype)

    src_x1 = max(0, x1_i)
    src_y1 = max(0, y1_i)
    src_x2 = min(w, x2_i)
    src_y2 = min(h, y2_i)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return Image.fromarray(out, mode=mode) if is_pil else out

    dst_x1 = src_x1 - x1_i
    dst_y1 = src_y1 - y1_i
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    out[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

    return Image.fromarray(out, mode=mode) if is_pil else out


def make_mso_views(image, mask: np.ndarray, bbox: Tuple[int, int, int, int], scale_factors=(2, 3, 4)):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    base = max(1.0, float(max(x2 - x1, y2 - y1)))

    views: List[Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]] = []

    for scale in scale_factors:
        crop_size = base * float(scale)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                shift_x = ox * crop_size / 3.0
                shift_y = oy * crop_size / 3.0
                center_x = cx + shift_x
                center_y = cy + shift_y
                crop_box = (
                    center_x - crop_size / 2.0,
                    center_y - crop_size / 2.0,
                    center_x + crop_size / 2.0,
                    center_y + crop_size / 2.0,
                )
                view_img = crop_with_padding(image, crop_box, fill=0)
                view_mask = crop_with_padding(mask, crop_box, fill=0)
                if isinstance(view_mask, Image.Image):
                    view_mask = np.array(view_mask)
                views.append((view_img, view_mask, crop_box))

    return views


def patch_mask_from_instance_mask(mask: np.ndarray, grid_size: Tuple[int, int], threshold: float = 0.10) -> np.ndarray:
    if mask.ndim != 2:
        mask = mask[..., 0]
    mask_float = (mask > 0).astype(np.float32)
    grid_h, grid_w = grid_size
    resized = cv2.resize(mask_float, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    patch_mask = resized >= float(threshold)
    return patch_mask.reshape(-1)
