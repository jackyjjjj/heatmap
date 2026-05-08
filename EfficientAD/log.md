# 更新日志

## 2026-05-08 +08:00

### 修改目的

将 EfficientAD 接入与 AnomalyDINO 类似的 MVTec-FS heatmap JSONL 输出模式，使 `work-1` 可以用 EfficientAD anomaly map 生成 pseudo-mask 并进行分类评估。

### 涉及文件

- `run_mvtecfs_efficientad.py`
- `efficientad.py`
- `log.md`

### 主要改动

- 新增 `run_mvtecfs_efficientad.py`，直接读取 MVTec-FS 的 `CONFIG/<object>_config1/<split>.csv`，按 object 输出 JSONL heatmap。
- 输出 schema 与 `work-1` heatmap 消费脚本兼容，包含图像路径、尺寸、heatmap 尺寸、heatmap 数组、localizer、model 和 score normalization 信息。
- 在 runner 内复用 EfficientAD 的 PDN teacher/student 和 autoencoder 结构，支持 `small` / `medium` 两种 model size。
- 默认 teacher 权重为 `models/teacher_<model_size>.pth`，per-object student/autoencoder 默认路径为 `output/1/trainings/mvtec_ad/<object>/student_final.pth` 和 `autoencoder_final.pth`。
- 支持 `--student-template`、`--autoencoder-template`、`--teacher-template` 改写权重路径，方便接入已有训练产物。
- 支持 `--normalize efficientad-quantile|per-image-minmax|none`，默认先用 train split 计算 teacher normalization 和 EfficientAD quantile map normalization。
- `efficientad.py` 保持 `if __name__ == '__main__': main()` 保护，方便被其他脚本安全导入或静态检查。

### 推荐运行方式

```bash
cd /home/jack/workspace/work-1
PROJECT=efficientad MODEL_SIZE=small RUN_ENV=AnomalyDINO bash run_heatmap_to_work1.sh
```

也可以单独运行 EfficientAD heatmap 导出：

```bash
cd /home/jack/workspace/heatmap/EfficientAD
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python run_mvtecfs_efficientad.py \
  --data_root /home/jack/workspace/data/MVTec-FS \
  --output_dir /home/jack/workspace/work-1/outputs/heatmaps/efficientad_raw \
  --objects bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper \
  --split valid --normalization-split train --model-size small --device cuda:0
```

### 验证

```bash
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python -m py_compile run_mvtecfs_efficientad.py efficientad.py
/home/jack/miniconda3/bin/conda run -n AnomalyDINO python run_mvtecfs_efficientad.py --help
```

结果：通过。完整推理需要先准备每个 object 的 EfficientAD student 和 autoencoder 训练权重。
## 2026-05-09 +08:00

- Integrated through work-1 batch script, but local per-object student_final.pth / autoencoder_final.pth files are missing so full heatmap generation was skipped in the latest batch run.
- Batch entry: `/home/jack/workspace/work-1/run_three_heatmap_models.sh`.
- Unified bridge: `/home/jack/workspace/work-1/run_heatmap_to_work1.sh`.
