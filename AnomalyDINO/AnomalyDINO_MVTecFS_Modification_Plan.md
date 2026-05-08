# AnomalyDINO 适配 MVTec-FS 修改计划

本文档用于交给代码 agent 直接修改 AnomalyDINO 代码，使其能够按 MVREC 论文的 few-shot 训练/评估方式在 MVTec-FS 数据集上运行，并在推理时输出每张图像一行的 heatmap JSONL 文件。

## 1. 目标

### 1.1 功能目标

将 AnomalyDINO 从原始 few-shot anomaly detection 流程改造成支持 MVTec-FS 的 few-shot defect multi-classification 流程：

- 每个 `object_name` 单独运行一次实验，例如 `capsule`、`bottle`、`cable`。
- 每个 `object_name` 内做 N-way K-shot 缺陷分类。
- support set 从 MVTec-FS 的 `CONFIG/{object_name}_config1/train.csv` 中按每个缺陷类别抽 K 张。
- query set 使用 `CONFIG/{object_name}_config1/valid.csv` 中所有样本。
- DINOv2 backbone 冻结，不从零训练。
- 训练/适配方式与 MVREC 保持一致：每个 object、shot、seed 单独建立 support cache；如实现训练版，则只训练 adapter/cache，不训练 DINOv2。
- 推理时为每张图像输出一行 JSON object 到 `.jsonl` 文件。
- heatmap 输出为 patch-level 分数矩阵，例如 `dinov2_vits14 + image_size=518` 时输出 `37 x 37`。

### 1.2 输出 JSONL 格式

每一行是一个 image 的 JSON object，字段至少包括：

```json
{
  "image_path": "capsule/train/crack/000.png",
  "label": "crack",
  "split": "train",
  "object_name": "capsule",
  "defect_name": "crack",
  "image_width": 1024,
  "image_height": 768,
  "heatmap_width": 37,
  "heatmap_height": 37,
  "heatmap": [
    [0.02, 0.04, 0.11],
    [0.08, 0.91, 0.55],
    [0.01, 0.30, 0.22]
  ],
  "localizer": "dinov2_patch-contrast",
  "model": "dinov2_vits14",
  "image_size": 518,
  "score_normalization": "per_image_minmax"
}
```

建议额外增加以下字段，方便调试和统计分类结果：

```json
{
  "pred_label": "crack",
  "class_scores": {
    "crack": 0.91,
    "poke": 0.22
  },
  "shot": 5,
  "seed": 0
}
```

## 2. 关键设计

### 2.1 与原始 AnomalyDINO 的差异

原始 AnomalyDINO 主要做：

```text
normal support images -> normal patch memory bank
query patch 与 normal memory 的距离 -> anomaly heatmap
```

MVTec-FS 适配后建议做：

```text
defect support images + defect mask/bbox -> 每个 defect class 的 patch memory bank
query patch 与各 defect class memory 的相似度 -> class score + heatmap
```

因此 heatmap 语义从“离 normal 越远越异常”变成：

```text
与某个 defect support patch 越相似，heatmap 分数越高
```

### 2.2 与 MVREC 训练协议对齐

实验循环应为：

```text
for object_name in objects:
    读取 train.csv 和 valid.csv
    for shot in [1, 3, 5]:
        for seed in [0, 1, 2, 3, 4]:
            每个 defect class 从 train.csv 抽 shot 张 support
            使用 support 建立/训练当前 object 的 few-shot classifier
            使用 valid.csv 全部样本做 query
            输出 jsonl 和 metrics
```

注意：

- 不是每个 defect class 一个模型。
- 是每个 object dataset 一个 N-way classifier/cache。
- `bottle` 是 3-way，`cable` 是 7-way，`capsule` 是 5-way，依此类推。

### 2.3 推荐优先级

建议分两阶段实现：

1. **第一阶段：无梯度 memory-bank 版**
   - 最贴近 AnomalyDINO 原始实现。
   - DINOv2 冻结。
   - support patch features 构建 per-class memory bank。
   - query patch 与 per-class memory 做 cosine similarity / FAISS inner product。
   - 可直接输出 heatmap JSONL。

2. **第二阶段：Zip-Adapter-F 训练版**
   - 更贴近 MVREC 的训练版。
   - DINOv2 仍然冻结。
   - 在 support image-level ROI feature 上训练 adapter/cache。
   - 训练参数建议与 MVREC 一致：AdamW、lr `1e-4`、500 iterations、CE loss + 4 * triplet loss、margin 0.5。
   - heatmap 仍由 patch-contrast localizer 输出；adapter 主要负责 `pred_label`。

## 3. 预期目录结构

在 AnomalyDINO 仓库中新增/修改如下文件：

```text
AnomalyDINO/
  run_mvtecfs_anomalydino.py
  src/
    backbones.py                 # 修改
    mvtec_fs.py                  # 新增
    mvtecfs_views.py             # 新增
    detection_mvtecfs.py          # 新增
    zip_adapter_dino.py           # 可选，第二阶段新增
  requirements.txt               # 如缺依赖则修改
```

## 4. 修改 `src/backbones.py`

### 4.1 目标

确保 DINOv2 支持固定正方形 resize，以得到稳定的 heatmap 尺寸。

对于 DINOv2 patch size 14：

```text
image_size = 518
heatmap_size = 518 / 14 = 37
```

因此要支持将输入图像强制 resize 到 `518 x 518`，而不是只 resize shorter edge。

### 4.2 需要新增参数

在 DINOv2 wrapper 和 `get_model()` 中新增：

```python
square_resize: bool = False
```

### 4.3 修改逻辑

如果 `square_resize=True`：

```python
transforms.Resize(
    (smaller_edge_size, smaller_edge_size),
    interpolation=transforms.InterpolationMode.BICUBIC,
    antialias=True,
)
```

否则保留原始 AnomalyDINO 行为：

```python
transforms.Resize(
    size=smaller_edge_size,
    interpolation=transforms.InterpolationMode.BICUBIC,
    antialias=True,
)
```

### 4.4 对外接口要求

确保模型 wrapper 至少提供以下能力：

```python
tensor, grid_size = model.prepare_image(pil_image)
features = model.extract_features(tensor)
```

其中：

- `grid_size` 应为 `(heatmap_height, heatmap_width)`。
- `features` 应为 shape `[num_patches, dim]` 的 numpy array 或 torch tensor。
- 若原始实现已经有类似接口，则不要重复造接口，只需要适配调用方。

## 5. 新增 `src/mvtec_fs.py`

### 5.1 目标

读取 MVTec-FS 数据集，统一输出 instance-level item。

MVTec-FS 预期结构：

```text
MVTec-FS/
  image/
    capsule/
      train/
      valid/
      ...
  CONFIG/
    capsule_config1/
      train.csv
      valid.csv
    bottle_config1/
      train.csv
      valid.csv
```

### 5.2 新增数据结构

建议定义：

```python
from dataclasses import dataclass

@dataclass
class MVTecFSItem:
    image_path: str
    rel_image_path: str
    label: str
    split: str
    object_name: str
    defect_name: str
    bbox: tuple
    points: list | None
    image_width: int
    image_height: int
    instance_id: str | None = None
```

### 5.3 新增 Dataset 类

建议实现：

```python
class MVTecFSInstanceDataset:
    def __init__(self, root, object_name, split):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

支持：

```python
train_ds = MVTecFSInstanceDataset(data_root, "capsule", "train")
valid_ds = MVTecFSInstanceDataset(data_root, "capsule", "valid")
```

### 5.4 CSV 字段兼容

不同版本 CSV 字段名可能不完全一致，读取时做兼容：

image path 字段候选：

```python
["img_rel_path", "imagePath", "path", "image_path"]
```

label 字段候选：

```python
["label", "class", "defect_name", "y"]
```

bbox 字段候选：

```python
["x1", "y1", "x2", "y2"]
```

points 字段候选：

```python
["points", "polygon"]
```

如 CSV 中没有 bbox，但有 LabelMe JSON，则从 JSON 中读取 `shapes`，按当前 label 找对应 shape，再由 polygon points 计算 bbox。

### 5.5 few-shot support 采样

新增函数：

```python
def sample_support_by_class(train_dataset, k_shot, seed):
    ...
```

要求：

- 按 `item.label` 分组。
- 每个 label 随机抽 `k_shot` 个样本。
- 使用固定 `seed` 保证可复现。
- 如果某类样本数小于 `k_shot`，直接报错。

## 6. 新增 `src/mvtecfs_views.py`

### 6.1 目标

实现 MVREC 风格的 region-context 多视角构造。

### 6.2 必须函数

#### `polygon_to_mask`

将 LabelMe polygon points 转成二值 mask：

```python
def polygon_to_mask(points, width, height):
    ...
```

输出：

```text
uint8 mask, shape [height, width], 0/255
```

#### `crop_with_padding`

支持越界 crop，并自动 padding：

```python
def crop_with_padding(img, box, fill=0):
    ...
```

要求：

- 输入可以是 PIL Image，也可以是 numpy mask。
- box 格式为 `(x1, y1, x2, y2)`。
- 越界区域用 `fill` 填充。

#### `make_mso_views`

实现 multi-scale + offset：

```python
def make_mso_views(image, mask, bbox, scale_factors=(2, 3, 4)):
    ...
```

建议逻辑：

```text
缺陷 bbox 中心为 cx, cy
base = max(bbox_width, bbox_height)
for scale in [2, 3, 4]:
    crop_size = base * scale
    for oy in [-1, 0, 1]:
        for ox in [-1, 0, 1]:
            crop_center = (cx + ox * crop_size / 3, cy + oy * crop_size / 3)
            生成 crop
```

总共输出 27 个 view：

```text
3 scales x 3 x-offsets x 3 y-offsets = 27 views
```

返回：

```python
[(view_img, view_mask, crop_box), ...]
```

#### `patch_mask_from_instance_mask`

将 instance mask resize 到 DINO patch grid：

```python
def patch_mask_from_instance_mask(mask, grid_size, threshold=0.10):
    ...
```

输出：

```text
bool array, shape [num_patches]
```

用途：

- support 阶段只取缺陷区域 patch feature。
- 如果 mask 覆盖比例大于 threshold，则该 patch 被视为 defect patch。

## 7. 新增 `src/detection_mvtecfs.py`

### 7.1 目标

实现 MVTec-FS 上的 AnomalyDINO patch-contrast 推理和 JSONL 输出。

### 7.2 新增 memory bank

建议新增：

```python
class ClassMemoryBank:
    def __init__(self, dim):
        ...

    def add(self, label, features):
        ...

    def build(self):
        ...

    def search_label(self, label, query_features, k=1):
        ...
```

实现要求：

- 每个 defect label 一个 memory bank。
- feature 先 L2 normalize。
- 使用 FAISS `IndexFlatIP` 或 torch cosine similarity。
- `search_label()` 返回每个 query patch 与当前 label support memory 的相似度。

如果项目不方便引入 `faiss`，可先用 torch 实现：

```python
sim = query @ memory.T
topk = sim.topk(k, dim=1).values
patch_scores = topk.mean(dim=1)
```

### 7.3 support feature 提取

新增：

```python
def extract_support_defect_patches(model, item, image_size, mv_method="mso"):
    ...
```

流程：

```text
读取原图
由 polygon points 生成 mask
如果 mv_method == "mso"，生成 27 个 region-context views
对每个 view:
    DINOv2 提取 patch feature
    mask resize 到 patch grid
    只保留 defect patch features
concat 所有 view 的 defect patch features
```

### 7.4 构建 support bank

新增：

```python
def build_mvtecfs_memory_bank(model, support_items, image_size, mv_method="mso"):
    ...
```

流程：

```text
for support item:
    features = extract_support_defect_patches(...)
    bank.add(item.label, features)
bank.build()
return bank
```

### 7.5 query 推理与 heatmap

新增：

```python
def infer_one_heatmap(model, bank, item, k_neighbors=1, heatmap_class="pred"):
    ...
```

流程：

```text
读取 query 原图
整图 resize 到 image_size x image_size
DINOv2 提取全部 patch features
for 每个 defect label:
    patch_scores = 当前 query patch 与该 label memory 的相似度
    score_map = patch_scores.reshape(grid_size)
    class_score = mean_top1p(score_map) 或 top-k mean
pred_label = class_score 最大的 label
如果 heatmap_class == "pred":
    输出 pred_label 的 score_map
如果 heatmap_class == "gt":
    输出 ground-truth label 的 score_map
score_map 做 per-image minmax
```

建议 `class_score` 默认使用 AnomalyDINO 原实现中的 `mean_top1p`。如果该函数位置不同，则复用原项目实现；没有则新增：

```python
def mean_top1p(scores):
    flat = np.asarray(scores).reshape(-1)
    k = max(1, int(len(flat) * 0.01))
    return float(np.sort(flat)[-k:].mean())
```

### 7.6 JSONL writer

新增：

```python
def write_heatmap_record(
    fp,
    item,
    heatmap,
    pred_label,
    class_scores,
    model_name,
    image_size,
    localizer="dinov2_patch-contrast",
    shot=None,
    seed=None,
):
    ...
```

要求：

- `heatmap` 转成 list。
- 建议保留 6 位小数，减少文件体积：

```python
np.round(heatmap, 6).tolist()
```

- 使用：

```python
json.dumps(record, ensure_ascii=False)
```

### 7.7 批量运行函数

新增：

```python
def run_mvtecfs_inference_jsonl(
    model,
    support_items,
    query_items,
    output_jsonl,
    model_name,
    image_size,
    mv_method="mso",
    k_neighbors=1,
    heatmap_class="pred",
    shot=None,
    seed=None,
):
    ...
```

返回：

```python
{
    "accuracy": 0.0,
    "correct": 0,
    "total": 0,
    "jsonl": "path/to/output.jsonl"
}
```

## 8. 新增 `run_mvtecfs_anomalydino.py`

### 8.1 目标

提供一个完整 CLI 入口，直接按 MVREC 协议跑 MVTec-FS。

### 8.2 参数设计

建议支持：

```bash
python run_mvtecfs_anomalydino.py \
  --data_root /path/to/MVTec-FS \
  --output_dir results_mvtecfs_anomalydino \
  --objects capsule bottle \
  --shots 1 3 5 \
  --num_seeds 5 \
  --model_name dinov2_vits14 \
  --image_size 518 \
  --device cuda:0 \
  --mv_method mso \
  --k_neighbors 1 \
  --heatmap_class pred
```

参数说明：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data_root` | 必填 | MVTec-FS 根目录 |
| `--output_dir` | `results_mvtecfs_anomalydino` | 输出目录 |
| `--objects` | 14 个 object | 要运行的 object_name |
| `--shots` | `1 3 5` | K-shot 设置 |
| `--num_seeds` | `5` | 每个 shot 重复采样次数 |
| `--model_name` | `dinov2_vits14` | DINOv2 模型 |
| `--image_size` | `518` | 输入尺寸 |
| `--device` | `cuda:0` | 运行设备 |
| `--mv_method` | `mso` | 多视角方式，支持 `none` / `mso` |
| `--k_neighbors` | `1` | patch memory 检索 top-k |
| `--heatmap_class` | `pred` | 输出预测类还是 GT 类的 heatmap |

### 8.3 默认 object 列表

使用 MVTec-FS/MVREC 中参与实验的 14 个 object：

```python
DEFAULT_OBJECTS = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "transistor", "wood", "zipper",
]
```

不建议默认包含 `toothbrush`，因为 MVREC 论文说明该 object 只有一种缺陷类型，通常不参与 N-way defect classification。

### 8.4 主循环

主函数逻辑：

```python
model = get_model(
    args.model_name,
    args.device,
    smaller_edge_size=args.image_size,
    square_resize=True,
)

for object_name in args.objects:
    train_ds = MVTecFSInstanceDataset(args.data_root, object_name, "train")
    valid_ds = MVTecFSInstanceDataset(args.data_root, object_name, "valid")

    for shot in args.shots:
        for seed in range(args.num_seeds):
            support_items = sample_support_by_class(train_ds, shot, seed)
            query_items = valid_ds.items

            output_jsonl = (
                Path(args.output_dir)
                / args.model_name
                / f"{object_name}_{shot}shot_seed{seed}.jsonl"
            )

            result = run_mvtecfs_inference_jsonl(...)
            print(result)
```

最后写一个汇总文件：

```text
results_mvtecfs_anomalydino/
  dinov2_vits14/
    capsule_1shot_seed0.jsonl
    capsule_1shot_seed1.jsonl
    capsule_3shot_seed0.jsonl
    metrics.json
```

## 9. 可选：新增 `src/zip_adapter_dino.py`

如果需要更严格对齐 MVREC 的训练版 Zip-Adapter-F，则新增该模块。

### 9.1 目标

在 DINOv2 ROI/multi-view image-level feature 上训练一个小 adapter/cache classifier。

### 9.2 输入特征

对每个 support item：

```text
使用 mso 生成 27 个 views
每个 view 提取 DINO patch features
只聚合 mask 覆盖的 defect patch features
对 patch features mean pooling
再对 27 个 view mean pooling
得到一个 item-level ROI feature
```

得到：

```python
support_features: [N_support, D]
support_labels: [N_support]
```

query item 同理得到：

```python
query_features: [N_query, D]
```

### 9.3 模型结构建议

最小实现：

```python
class ZipAdapterDINO(nn.Module):
    def __init__(self, dim, num_classes, support_features, support_labels):
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.cache_keys = nn.Parameter(support_features.clone())
        self.cache_values = one_hot(support_labels, num_classes)

    def forward(self, x):
        q = F.normalize(self.proj(x), dim=-1)
        k = F.normalize(self.proj(self.cache_keys), dim=-1)
        logits_cache = beta * q @ k.T
        logits = logits_cache @ self.cache_values
        return logits
```

### 9.4 训练配置

与 MVREC 对齐：

```text
optimizer: AdamW
learning_rate: 1e-4
iterations: 500
loss: CE loss + 4.0 * triplet loss
triplet_margin: 0.5
backbone: frozen
```

注意：few-shot setting 中 query set 不应参与训练。训练只使用 support features。由于 support 极少，训练时可以对 support features 做轻量增强，例如 dropout/noise，但要清楚记录。

### 9.5 与 heatmap 的关系

Zip-Adapter-DINO 主要用于：

```text
pred_label
class_scores
classification accuracy
```

heatmap 仍建议用 `detection_mvtecfs.py` 中的 patch-contrast memory bank 输出。

## 10. 依赖修改

检查 `requirements.txt` 是否包含：

```text
numpy
Pillow
tqdm
opencv-python
torch
torchvision
```

FAISS 可选：

```text
faiss-cpu
```

如果用户在 Windows 上运行，优先使用 `faiss-cpu` 或 torch fallback，不要强依赖 `faiss-gpu`。

## 11. 验证步骤

### 11.1 最小 smoke test

先只跑一个 object、一个 shot、一个 seed：

```powershell
python run_mvtecfs_anomalydino.py `
  --data_root D:\datasets\MVTec-FS `
  --output_dir results_mvtecfs_anomalydino `
  --objects capsule `
  --shots 1 `
  --num_seeds 1 `
  --model_name dinov2_vits14 `
  --image_size 518 `
  --device cuda:0 `
  --mv_method mso `
  --k_neighbors 1 `
  --heatmap_class pred
```

如果没有 GPU，改成：

```powershell
--device cpu
```

### 11.2 检查输出文件

确认存在：

```text
results_mvtecfs_anomalydino/
  dinov2_vits14/
    capsule_1shot_seed0.jsonl
    metrics.json
```

检查 JSONL 第一行：

```python
import json

path = "results_mvtecfs_anomalydino/dinov2_vits14/capsule_1shot_seed0.jsonl"
with open(path, "r", encoding="utf-8") as f:
    obj = json.loads(next(f))

assert obj["object_name"] == "capsule"
assert obj["heatmap_width"] == 37
assert obj["heatmap_height"] == 37
assert len(obj["heatmap"]) == 37
assert len(obj["heatmap"][0]) == 37
```

### 11.3 指标检查

`metrics.json` 中每个实验至少应有：

```json
{
  "object_name": "capsule",
  "shot": 1,
  "seed": 0,
  "accuracy": 0.0,
  "correct": 0,
  "total": 0,
  "jsonl": "..."
}
```

### 11.4 多 object 测试

smoke test 通过后再跑：

```powershell
python run_mvtecfs_anomalydino.py `
  --data_root D:\datasets\MVTec-FS `
  --output_dir results_mvtecfs_anomalydino `
  --objects bottle capsule transistor `
  --shots 1 3 5 `
  --num_seeds 5 `
  --model_name dinov2_vits14 `
  --image_size 518 `
  --device cuda:0 `
  --mv_method mso
```

## 12. 常见风险和处理方案

### 12.1 CSV 字段不匹配

风险：

```text
MVTec-FS 的 CSV 字段名可能和预期不一致。
```

处理：

- 先打印 `reader.fieldnames`。
- 在 `mvtec_fs.py` 中增加字段候选。
- 如果 CSV 只存 JSON 路径，则从 JSON 中读取 image path、label、points。

### 12.2 LabelMe JSON 中一个图有多个实例

风险：

```text
同一张图可能有多个 defect instance 或多个 shape。
```

处理：

- Dataset item 应按 instance 组织，而不是单纯按 image 组织。
- 如果 CSV 已经一行一个 instance，则按 CSV 为准。
- 如果 CSV 一行一图，但 JSON 中多 shape，则需要展开为多条 `MVTecFSItem`。

### 12.3 heatmap 尺寸不是 37x37

原因：

- 没有使用 `square_resize=True`。
- 输入尺寸不是 518。
- DINO 模型 patch size 不是 14。

处理：

- 强制 `Resize((518, 518))`。
- 确认 `grid_size == (37, 37)`。

### 12.4 FAISS 安装失败

处理：

- 不要强依赖 FAISS。
- 在 `detection_mvtecfs.py` 中提供 torch fallback。

### 12.5 推理速度慢

原因：

- `mso` 每个 support instance 要提 27 次 DINO feature。

处理：

- 先支持 `--mv_method none` 做 debug。
- 后续可缓存 support features 到 `.npz`。
- 可增加参数 `--cache_dir`。

## 13. 建议交付标准

代码 agent 修改完成后，应交付：

- 新增文件：
  - `run_mvtecfs_anomalydino.py`
  - `src/mvtec_fs.py`
  - `src/mvtecfs_views.py`
  - `src/detection_mvtecfs.py`
  - 可选：`src/zip_adapter_dino.py`
- 修改文件：
  - `src/backbones.py`
  - 可选：`requirements.txt`
- 可运行命令：
  - 至少能跑通 `capsule 1-shot seed0`。
- 输出验证：
  - JSONL 每行合法 JSON。
  - `heatmap_width == 37`。
  - `heatmap_height == 37`。
  - `metrics.json` 包含 accuracy/correct/total。

## 14. 推荐首个运行命令

```powershell
python run_mvtecfs_anomalydino.py `
  --data_root D:\datasets\MVTec-FS `
  --output_dir results_mvtecfs_anomalydino `
  --objects capsule `
  --shots 1 `
  --num_seeds 1 `
  --model_name dinov2_vits14 `
  --image_size 518 `
  --device cuda:0 `
  --mv_method none `
  --k_neighbors 1 `
  --heatmap_class pred
```

确认 `none` 模式跑通后，再改成：

```powershell
--mv_method mso
```

## 15. 最终实验命令示例

```powershell
python run_mvtecfs_anomalydino.py `
  --data_root D:\datasets\MVTec-FS `
  --output_dir results_mvtecfs_anomalydino `
  --objects bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper `
  --shots 1 3 5 `
  --num_seeds 5 `
  --model_name dinov2_vits14 `
  --image_size 518 `
  --device cuda:0 `
  --mv_method mso `
  --k_neighbors 1 `
  --heatmap_class pred
```

