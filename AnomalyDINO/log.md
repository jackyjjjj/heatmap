# 更新日志

## 2026-05-08 +08:00

### 修改目的

让 AnomalyDINO 在 WSL workspace 下稳定为 MVTec-FS 输出 `work-1` 可消费的 heatmap JSONL，并避免 `torch.hub` 访问 GitHub API 时触发 rate limit 导致 DINOv2 加载失败。

### 涉及文件

- `src/backbones.py`
- `run_mvtecfs_anomalydino.py`
- `log.md`

### 主要改动

- 更新 `src/backbones.py` 的 `DINOv2Wrapper.load_model()`：新增 `DINOV2_REPO_OR_DIR` 环境变量支持，可直接指向本地 DINOv2 repo/cache。
- 根据 `DINOV2_REPO_OR_DIR` 是否为本地目录自动选择 `torch.hub.load(..., source="local")` 或 `source="github"`。
- 对 GitHub source 增加 `trust_repo=True` 和 `skip_validation=True`，规避 GitHub API repo validation 的 rate limit 问题，避免出现 `HTTP Error 403: rate limit exceeded` 后继续触发 `KeyError: 'Authorization'`。
- 使用 `run_mvtecfs_anomalydino.py` 在 MVTec-FS 上按 object 输出 heatmap JSONL，输出目录交由 `work-1/run_anomalydino_to_work1.sh` 指向 `work-1/outputs/heatmaps/anomalydino_raw`。

### 运行配置

- 数据集：`/home/jack/workspace/data/MVTec-FS`
- 输出目录：`/home/jack/workspace/work-1/outputs/heatmaps/anomalydino_raw`
- 模型：`dinov2_vits14`
- 图像尺寸：`518`
- 设备：`cuda:0`
- few-shot：`shots=1`，`num_seeds=1`，`seed=0`
- heatmap：`mv_method=mso`，`k_neighbors=1`，`heatmap_class=gt`
- 本地 DINOv2 cache：`/home/jack/.cache/torch/hub/facebookresearch_dinov2_main`
- 本地 DINOv2 weight：`/home/jack/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`

### 验证结果

- AnomalyDINO 已成功生成各 object 的 `*_1shot_seed0.jsonl` heatmap 文件。
- `work-1` 侧合并后得到 720 行 heatmap，按 `image_path` 像素级 `max` 去重后得到 594 行，与 `mvtec_fs.csv` 的 `test=594` 对齐。
- 去重 heatmap 已成功用于生成 pseudo-bbox 和 pseudo-mask，并完成后续 DINOv2 特征提取与 few-shot 分类评估。

### 备注

当前仓库中 `__pycache__/` 为运行 Python 后产生的缓存目录；本次记录未对该目录做清理或提交处理。