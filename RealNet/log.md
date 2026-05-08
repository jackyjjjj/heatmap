# 更新日志

## 2026-05-08 +08:00

### 修改目的

将 RealNet 接入与 AnomalyDINO 类似的 MVTec-FS heatmap JSONL 输出模式，使 `work-1` 可以直接消费 RealNet anomaly score map 并生成 pseudo-mask / 分类评估结果。

### 涉及文件

- `run_mvtecfs_realnet.py`
- `log.md`

### 主要改动

- 新增 `run_mvtecfs_realnet.py`，直接读取 MVTec-FS 的 `CONFIG/<object>_config1/<split>.csv`，按 object 输出 JSONL heatmap。
- 输出 schema 与 `work-1/scripts/build_pseudo_mask_manifest.py` 兼容，包含 `image_path`、`label`、`split`、`object_name`、`defect_name`、`image_width`、`image_height`、`heatmap_width`、`heatmap_height`、`heatmap` 等字段。
- 默认读取 `experiments/MVTec-AD/realnet.yaml` 构建 RealNet，并用 `experiments/MVTec-AD/realnet_checkpoints/<object>/ckpt_best.pth.tar` 加载每个 object 的 checkpoint。
- 支持 `--checkpoint-template` 和 `--checkpoint-root`，方便 checkpoint 不在默认目录时改路径。
- 支持 `--heatmap-resolution model|image` 和 `--normalize per-image-minmax|none`，默认输出 model 分辨率热图并做逐图 min-max 归一化；`work-1` 后续可用 `--upsample-heatmap-to-image` 上采样到原图尺寸。

### 推荐运行方式

```bash
cd /home/jack/workspace/work-1
PROJECT=realnet RUN_ENV=AnomalyDINO bash run_heatmap_to_work1.sh
```

也可以单独运行 RealNet heatmap 导出：

```bash
cd /home/jack/workspace/heatmap/RealNet
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python run_mvtecfs_realnet.py \
  --data_root /home/jack/workspace/data/MVTec-FS \
  --output_dir /home/jack/workspace/work-1/outputs/heatmaps/realnet_raw \
  --objects bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper \
  --split valid --device cuda:0 --localizer realnet
```

### 验证

```bash
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python -m py_compile run_mvtecfs_realnet.py
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python run_mvtecfs_realnet.py --help
```

结果：通过。完整推理需要先准备每个 object 的 RealNet checkpoint。
## 2026-05-09 +08:00

- Integrated through work-1 batch script, but local per-object ckpt_best.pth.tar files are missing so full heatmap generation was skipped in the latest batch run.
- Batch entry: `/home/jack/workspace/work-1/run_three_heatmap_models.sh`.
- Unified bridge: `/home/jack/workspace/work-1/run_heatmap_to_work1.sh`.
