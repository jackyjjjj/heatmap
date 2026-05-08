#!/usr/bin/env bash
set -euo pipefail

# 1) 修改成你的数据根目录（MVTec-FS）
DATA_ROOT="/home/think/mnt/jyl/MyWork/data/MVTec-FS"

# 2) 输出目录（AnomalyDINO 会在这里写 jsonl）
OUTPUT_DIR="/home/think/mnt/jyl/MyWork/heatmap/AnomalyDINO/results_mvtecfs_anomalydino"

# 3) 设备（如果驱动老，改成 cpu）
DEVICE="cuda:0"        # 可改为 cuda:0
MODEL="dinov2_vits14"
IMAGE_SIZE=518

# 4) 目标 objects（可按需精简）
OBJECTS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper"

# 5) shots / seeds
SHOTS="1"
NUM_SEEDS=1

python run_mvtecfs_anomalydino.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --objects ${OBJECTS} \
  --shots ${SHOTS} \
  --num_seeds ${NUM_SEEDS} \
  --model_name "${MODEL}" \
  --image_size "${IMAGE_SIZE}" \
  --device "${DEVICE}" \
  --mv_method mso \
  --k_neighbors 1 \
  --heatmap_class pred