#!/bin/bash
# =============================================================================
# Cross-Embodiment Inference Script
#
# 输入一个机器人视频 + action序列，翻译成另一个机器人视频 + 新action序列
#
# 参数说明:
#   --wan_path:       Wan2.2-Fun-5B-Control 模型目录
#   --vae_path:       Wan2.2 VAE 权重路径
#   --source_video:   源机器人视频 (mp4)
#   --source_actions: 源机器人action序列 (.npy), shape [T_a, action_dim]
#   --output_video:   输出目标机器人视频路径
#   --output_actions: 输出目标机器人action序列路径
#   --checkpoint:     微调后的checkpoint路径
#   --num_frames:     视频帧数 (默认49)
#   --height:         视频高度 (默认480)
#   --width:          视频宽度 (默认832)
#   --fps:            视频帧率 (默认30)
#   --action_dim:     Action维度 (默认14)
#   --num_inference_steps: 去噪步数 (默认50)
#   --seed:           随机种子 (默认42)
# =============================================================================

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate motus

PROJECT_ROOT="/home/a100/youjingzhou/codes/Motus"
WAN_PATH="${PROJECT_ROOT}/pretrained_models/Wan2.2-Fun-5B-Control"
VAE_PATH="${WAN_PATH}/Wan2.2_VAE.pth"
CHECKPOINT="${PROJECT_ROOT}/pretrained_models/Motus/mp_rank_00_model_states.pt"

# T5 and CLIP model paths
T5_CHECKPOINT="${WAN_PATH}/models_t5_umt5-xxl-enc-bf16.pth"
T5_TOKENIZER="${WAN_PATH}/google/umt5-xxl"
CLIP_CHECKPOINT="/home/a100/youjingzhou/codes/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"

SOURCE_VIDEO="/home/a100/youjingzhou/data/all_data/test_data/wrist_generate/agibot_traj/control_video/black_49f_832x480.mp4"
SOURCE_ACTIONS="${PROJECT_ROOT}/test_actions.npy"

# Prompt and reference image
PROMPT="Pour water from kettle to flowers"
REFERENCE_IMAGE="/home/a100/youjingzhou/data/all_data/test_data/wrist_generate/agibot_traj/reference_image_masked/327/648642/2.jpg"

OUTPUT_VIDEO="output/target.mp4"
OUTPUT_ACTIONS="output/target_actions.npy"

NUM_FRAMES=49
HEIGHT=480
WIDTH=832
FPS=30
ACTION_DIM=14
NUM_INFERENCE_STEPS=50
SEED=42

cd "${PROJECT_ROOT}"

mkdir -p output

echo "============================================"
echo "Cross-Embodiment Inference"
echo "============================================"
echo "Source video:   ${SOURCE_VIDEO}"
echo "Source actions: ${SOURCE_ACTIONS}"
echo "Output video:   ${OUTPUT_VIDEO}"
echo "Output actions: ${OUTPUT_ACTIONS}"
echo "Video:          ${NUM_FRAMES}f @ ${FPS}fps, ${HEIGHT}x${WIDTH}"
echo "Action dim:     ${ACTION_DIM}"
echo "Inference steps: ${NUM_INFERENCE_STEPS}"
echo "============================================"

python inference/real_world/Motus/inference_cross_embodiment.py \
    --wan_path "${WAN_PATH}" \
    --vae_path "${VAE_PATH}" \
    --source_video "${SOURCE_VIDEO}" \
    --source_actions "${SOURCE_ACTIONS}" \
    --output_video "${OUTPUT_VIDEO}" \
    --output_actions "${OUTPUT_ACTIONS}" \
    --checkpoint "${CHECKPOINT}" \
    --num_frames ${NUM_FRAMES} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --fps ${FPS} \
    --action_dim ${ACTION_DIM} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --prompt "${PROMPT}" \
    --reference_image "${REFERENCE_IMAGE}" \
    --t5_checkpoint "${T5_CHECKPOINT}" \
    --t5_tokenizer "${T5_TOKENIZER}" \
    --clip_checkpoint "${CLIP_CHECKPOINT}"

echo ""
echo "Done! Output saved to:"
echo "  Video:   ${OUTPUT_VIDEO}"
echo "  Actions: ${OUTPUT_ACTIONS}"