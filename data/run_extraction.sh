#!/bin/bash

# ==============================================================================
# DUD-E 数据集特征提取流水线
# ==============================================================================

# 1. 定义你要跑的靶点 (一定要和 processed 目录下的文件夹名字一致！)
TARGET="lck"  

# 2. 定义路径
MODEL_WEIGHTS="/root/autodl-tmp/AI4S1/code_2/data/model_weights/6_folds/fold_0.pt"
DICT_DIR="/root/autodl-tmp/AI4S1/dict"
WORK_DIR="/root/autodl-tmp/AI4S1/DUD-E/processed/${TARGET}"
LABEL_TXT_FILE="${WORK_DIR}/${TARGET}_label.txt"
OUTPUT_DIR="${WORK_DIR}/features"

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "🚀 开始提取 DUD-E 靶点 [$TARGET] 的特征..."
echo "========================================================"

# --- 步骤 1：提取口袋特征 ---
echo -e "\n[1/2] 正在提取靶点口袋特征..."
python /root/autodl-tmp/AI4S1/code_2/encode_pocket_2.py \
    --path "$MODEL_WEIGHTS" \
    --label-file "$LABEL_TXT_FILE" \
    --batch-size 16 \
    --task drugclip --arch drugclip --fp16 \
    --results-path "$OUTPUT_DIR" \
    "$DICT_DIR"

# --- 步骤 2：提取分子库特征 (调用你发来的这份终极版代码) ---
echo -e "\n[2/2] 正在提取配体特征 (纯内存 Pickle 模式)..."
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/AI4S1/code_2/encode_ligand_3.py \
    --path "$MODEL_WEIGHTS" \
    --label-file "$LABEL_TXT_FILE" \
    --batch-size 512 \
    --task drugclip --arch drugclip --fp16 \
    --results-path "$OUTPUT_DIR" \
    "$DICT_DIR"

echo "========================================================"
echo "🏆 DUD-E $TARGET 特征提取完毕！"
echo "========================================================"