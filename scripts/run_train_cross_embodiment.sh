#!/bin/bash
# =============================================================================
# Cross-Embodiment Training Script
#
# 训练模型将源机器人视频+action翻译成目标机器人视频+action
# 基于 Flow Matching (SFT) 训练范式
#
# 训练数据格式 (JSONL metadata):
#   每行一个JSON对象:
#   {
#     "source_video": "relative/path/to/source.mp4",
#     "source_actions": "relative/path/to/source_actions.npy",
#     "target_video": "relative/path/to/target.mp4",
#     "target_actions": "relative/path/to/target_actions.npy",
#     "prompt": "optional text instruction",
#     "reference_image": "relative/path/to/reference_image.jpg"
#   }
#
# 参数说明:
#   --wan_path:                Wan2.2-Fun-5B-Control 模型目录
#   --vae_path:                Wan2.2 VAE 权重路径
#   --dataset_metadata_path:   训练数据metadata JSONL文件
#   --dataset_base_path:       数据集根目录
#   --output_path:             Checkpoint输出目录
#   --num_frames:              视频帧数 (默认49)
#   --height:                  视频高度 (默认480)
#   --width:                   视频宽度 (默认832)
#   --fps:                     视频帧率 (默认30)
#   --action_dim:              Action维度 (默认14)
#   --batch_size:              每GPU batch size (默认1)
#   --num_epochs:              训练epoch数 (默认100)
#   --lr:                      学习率 (默认1e-4)
#   --gradient_accumulation_steps: 梯度累积步数 (默认1)
#   --freeze_dit:              冻结DiT权重，仅训练action encoder/head
#   --trainable_models:        可训练的子模块列表
#   --t5_checkpoint:           T5文本编码器权重路径
#   --t5_tokenizer:            T5 tokenizer路径
#   --clip_checkpoint:         CLIP图像编码器权重路径
# =============================================================================

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate motus

PROJECT_ROOT="/home/a100/youjingzhou/codes/Motus"
WAN_PATH="${PROJECT_ROOT}/pretrained_models/Wan2.2-Fun-5B-Control"
VAE_PATH="${WAN_PATH}/Wan2.2_VAE.pth"

# T5 and CLIP model paths
T5_CHECKPOINT="${WAN_PATH}/models_t5_umt5-xxl-enc-bf16.pth"
T5_TOKENIZER="${WAN_PATH}/google/umt5-xxl"
CLIP_CHECKPOINT="/home/a100/youjingzhou/codes/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"

DATASET_METADATA="${PROJECT_ROOT}/data/train_metadata.jsonl"
DATASET_BASE="${PROJECT_ROOT}/data"

OUTPUT_PATH="${PROJECT_ROOT}/output/checkpoints"

NUM_FRAMES=49
HEIGHT=480
WIDTH=832
FPS=30
ACTION_DIM=14

BATCH_SIZE=1
NUM_EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-2
GRADIENT_ACCUMULATION_STEPS=1

NUM_WORKERS=4
LOG_INTERVAL=10
SAVE_INTERVAL=1000
SEED=42

FREEZE_DIT="--freeze_dit"
TRAINABLE_MODELS="action_control_encoder,action_head,fps_embedding"

cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_PATH}"

echo "============================================"
echo "Cross-Embodiment Training"
echo "============================================"
echo "WAN path:       ${WAN_PATH}"
echo "VAE path:       ${VAE_PATH}"
echo "Dataset:        ${DATASET_METADATA}"
echo "Output:         ${OUTPUT_PATH}"
echo "Video:          ${NUM_FRAMES}f @ ${FPS}fps, ${HEIGHT}x${WIDTH}"
echo "Action dim:     ${ACTION_DIM}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Epochs:         ${NUM_EPOCHS}"
echo "Learning rate:  ${LR}"
echo "Freeze DiT:     ${FREEZE_DIT}"
echo "Trainable:      ${TRAINABLE_MODELS}"
echo "============================================"

python scripts/train_cross_embodiment.py \
    --wan_path "${WAN_PATH}" \
    --vae_path "${VAE_PATH}" \
    --dataset_metadata_path "${DATASET_METADATA}" \
    --dataset_base_path "${DATASET_BASE}" \
    --output_path "${OUTPUT_PATH}" \
    --num_frames ${NUM_FRAMES} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --fps ${FPS} \
    --action_dim ${ACTION_DIM} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --seed ${SEED} \
    ${FREEZE_DIT} \
    --trainable_models "${TRAINABLE_MODELS}" \
    --t5_checkpoint "${T5_CHECKPOINT}" \
    --t5_tokenizer "${T5_TOKENIZER}" \
    --clip_checkpoint "${CLIP_CHECKPOINT}"

echo ""
echo "Training complete! Checkpoints saved to: ${OUTPUT_PATH}"