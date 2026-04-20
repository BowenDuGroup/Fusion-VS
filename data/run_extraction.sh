#!/bin/bash

# ==============================================================================
# Feature extraction pipeline for DUD-E dataset
# ==============================================================================

# 1. Define target (must match the folder name in 'processed')
TARGET="lck"  

# 2. Define paths
MODEL_WEIGHTS="/root/autodl-tmp/AI4S1/code_2/data/model_weights/6_folds/fold_0.pt"
DICT_DIR="/root/autodl-tmp/AI4S1/dict"
WORK_DIR="/root/autodl-tmp/AI4S1/DUD-E/processed/${TARGET}"
LABEL_TXT_FILE="${WORK_DIR}/${TARGET}_label.txt"
OUTPUT_DIR="${WORK_DIR}/features"

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting feature extraction for DUD-E target [$TARGET]..."
echo "========================================================"

# --- Step 1: Extract pocket features ---
echo -e "\n[1/2] Extracting pocket features..."
python /root/autodl-tmp/AI4S1/code_2/encode_pocket_2.py \
    --path "$MODEL_WEIGHTS" \
    --label-file "$LABEL_TXT_FILE" \
    --batch-size 16 \
    --task drugclip --arch drugclip --fp16 \
    --results-path "$OUTPUT_DIR" \
    "$DICT_DIR"

# --- Step 2: Extract ligand features (Pickle mode) ---
echo -e "\n[2/2] Extracting ligand features (In-memory Pickle mode)..."
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/AI4S1/code_2/encode_ligand_3.py \
    --path "$MODEL_WEIGHTS" \
    --label-file "$LABEL_TXT_FILE" \
    --batch-size 512 \
    --task drugclip --arch drugclip --fp16 \
    --results-path "$OUTPUT_DIR" \
    "$DICT_DIR"

echo "========================================================"
echo "Feature extraction for DUD-E $TARGET completed!"
echo "========================================================"