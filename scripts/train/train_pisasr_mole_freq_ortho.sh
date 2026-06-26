#!/bin/bash

# ============ Configurable Parameters ============
OUTPUT_DIR_BASE="experiments/mole-freq-ortho"
LEARNING_RATE=5e-5
TRAIN_BATCH_SIZE=4
PROB=0.1
SEED=123
LORA_RANK_UNET_PIX=4
LORA_RANK_UNET_SEM=4
NUM_EXPERTS_PIX=4
NUM_SHARED_EXPERTS_PIX=0
TOP_K_PIX=2
PIX_STEPS=4000
FREQ_DIM=7

# Ortho parameters
SVD_ENERGY_THRESHOLD=0.95

# ============ Auto-generate suffix ============
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUFFIX="_lr${LEARNING_RATE}_bs${TRAIN_BATCH_SIZE}_prob${PROB}_pix${LORA_RANK_UNET_PIX}_sem${LORA_RANK_UNET_SEM}_exp${NUM_EXPERTS_PIX}_topk${TOP_K_PIX}_shared${NUM_SHARED_EXPERTS_PIX}_freq${FREQ_DIM}_ortho${SVD_ENERGY_THRESHOLD}_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}${SUFFIX}"

echo "Output directory: ${OUTPUT_DIR}"

# ============ Training ============
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config.yml train_pisasr.py \
    --pretrained_model_path="preset/models/SD21" \
    --pretrained_model_path_csd="preset/models/SD21" \
    --dataset_txt_paths="preset/gt_path.txt" \
    --highquality_dataset_txt_paths="preset/gt_selected_path.txt" \
    --dataset_test_folder="preset/testfolder" \
    --learning_rate=${LEARNING_RATE} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --prob=${PROB} \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 500 \
    --seed ${SEED} \
    --output_dir="${OUTPUT_DIR}" \
    --cfg_csd 7.5 \
    --timesteps1 1 \
    --lambda_lpips=2.0 \
    --lambda_l2=1.0 \
    --lambda_csd=1.0 \
    --pix_steps=${PIX_STEPS} \
    --lora_rank_unet_pix=${LORA_RANK_UNET_PIX} \
    --lora_rank_unet_sem=${LORA_RANK_UNET_SEM} \
    --min_dm_step_ratio=0.02 \
    --max_dm_step_ratio=0.5 \
    --null_text_ratio=0.5 \
    --align_method="adain" \
    --deg_file_path="params.yml" \
    --tracker_project_name "PiSASR" \
    --is_module True \
    --num_experts_pix=${NUM_EXPERTS_PIX} \
    --top_k_pix=${TOP_K_PIX} \
    --num_shared_experts_pix=${NUM_SHARED_EXPERTS_PIX} \
    --use_load_balance_loss \
    --lambda_load_balance=0.01 \
    --use_freq_gate \
    --freq_dim=${FREQ_DIM} \
    --ortho_enabled \
    --svd_energy_threshold=${SVD_ENERGY_THRESHOLD} \
    --save_svd_subspaces
