#!/bin/bash

# ============ Evaluate Metrics Script ============
# Calculate PSNR, SSIM, LPIPS, DISTS, FID, CLIPIQA, NIQE, MUSIQ, MANIQA
#
# Usage:
#   ./eval_metrics.sh
#
# Configure the EVAL_LIST below with format:
#   "GT_TYPE:SR_DIR"
#
# GT_TYPE can be: DIV2K, RealSR, DRealSR (or add more in GT_DIRS)

# ============ Configurable Parameters ============
GPU_ID=4
LOG_DIR=experiments/logs

# ============ GT Directory Mapping ============
declare -A GT_DIRS
GT_DIRS["DIV2K"]="preset/DIV2K-Val/HR"
GT_DIRS["RealSR"]="preset/testfolder/test_HR"
GT_DIRS["DRealSR"]="preset/DRealSR/HR"

# ============ List of SR directories to evaluate ============
# Format: "GT_TYPE:SR_DIR"
EVAL_LIST=(
    "DIV2K:experiments/mole-freq-ortho_lr5e-5_bs4_prob0.1_pix4_sem4_exp8_topk2_shared0_freq7_ortho0.95_20260228_165306_model_30501_DIV2K-Val_default"
    "DRealSR:experiments/mole-freq-ortho_lr5e-5_bs4_prob0.1_pix4_sem4_exp8_topk2_shared0_freq7_ortho0.95_20260228_165306_model_30501_DRealSR_default"
    "RealSR:experiments/mole-freq-ortho_lr5e-5_bs4_prob0.1_pix4_sem4_exp8_topk2_shared0_freq7_ortho0.95_20260228_165306_model_30501_test_SR_bicubic_default"
)

# ============ Summary Results File ============
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/eval_summary_${TIMESTAMP}.txt"

# ============ Main Evaluation Loop ============
echo "=============================================="
echo "Batch Evaluation Started"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Total evaluations: ${#EVAL_LIST[@]}"
echo "Summary file: $SUMMARY_FILE"
echo "=============================================="

# Create log directory if not exists
mkdir -p "$LOG_DIR"

# Initialize summary file
echo "=============================================" > "$SUMMARY_FILE"
echo "Evaluation Summary - $(date)" >> "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Counter
COUNT=0
TOTAL=${#EVAL_LIST[@]}

for ENTRY in "${EVAL_LIST[@]}"; do
    COUNT=$((COUNT + 1))

    # Parse GT_TYPE and SR_DIR
    GT_TYPE="${ENTRY%%:*}"
    SR_DIR="${ENTRY#*:}"

    # Get GT_DIR from mapping
    GT_DIR="${GT_DIRS[$GT_TYPE]}"

    if [ -z "$GT_DIR" ]; then
        echo "[ERROR] Unknown GT_TYPE: $GT_TYPE"
        echo "[ERROR] Unknown GT_TYPE: $GT_TYPE for $SR_DIR" >> "$SUMMARY_FILE"
        continue
    fi

    # Extract experiment name from SR_DIR
    EXP_NAME=$(basename "$SR_DIR")

    echo ""
    echo "=============================================="
    echo "[$COUNT/$TOTAL] Evaluating: $EXP_NAME"
    echo "=============================================="
    echo "GT Type: $GT_TYPE"
    echo "SR images: $SR_DIR"
    echo "GT images: $GT_DIR"
    echo "=============================================="

    # Write to summary
    echo "----------------------------------------" >> "$SUMMARY_FILE"
    echo "[$COUNT/$TOTAL] $EXP_NAME" >> "$SUMMARY_FILE"
    echo "GT Type: $GT_TYPE" >> "$SUMMARY_FILE"
    echo "SR: $SR_DIR" >> "$SUMMARY_FILE"
    echo "GT: $GT_DIR" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    # Run evaluation and capture output
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_metrics.py \
        --inp_imgs "$SR_DIR" \
        --gt_imgs "$GT_DIR" \
        --log "$LOG_DIR" 2>&1 | tee -a "$SUMMARY_FILE"

    echo "" >> "$SUMMARY_FILE"
done

echo ""
echo "=============================================="
echo "Batch Evaluation Completed"
echo "=============================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "=============================================="
