#!/bin/bash

# ============ Configurable Parameters ============
GPU_ID=1

# Path to the trained checkpoint
PRETRAINED_PATH=experiments/train-pisasr_mole-freq-ortho_lr5e-5_bs8_prob0.1_pix4_sem4_exp4_topk2_freq7_ortho0.95_20260129_001559/checkpoints/model_30501.pkl

# Input image or folder
INPUT_IMAGE=preset/kisf-sr_org_part2

# ============ Auto generate output_dir from pretrained_path ============
# e.g., .../train-pisasr_prob0.1/checkpoints/model_12501.pkl => train-pisasr_prob0.1_model_12501
EXP_NAME=$(basename $(dirname $(dirname $PRETRAINED_PATH)))
MODEL_NAME=$(basename $PRETRAINED_PATH .pkl)
# Get dataset name: if folder ends with LR/HR, use parent; otherwise use folder name itself
INPUT_BASENAME=$(basename $INPUT_IMAGE)
if [[ "$INPUT_BASENAME" == "LR" || "$INPUT_BASENAME" == "HR" ]]; then
    DATASET_NAME=$(basename $(dirname $INPUT_IMAGE))
else
    DATASET_NAME=$INPUT_BASENAME
fi
OUTPUT_DIR="experiments/${EXP_NAME}_${MODEL_NAME}_${DATASET_NAME}_default"

echo "=============================================="
echo "Testing Model"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Checkpoint: $PRETRAINED_PATH"
echo "Input: $INPUT_IMAGE"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python test_pisasr.py \
--pretrained_model_path preset/models/SD21 \
--pretrained_path $PRETRAINED_PATH \
--process_size 512 \
--upscale 4 \
--input_image $INPUT_IMAGE \
--output_dir $OUTPUT_DIR \
--default

