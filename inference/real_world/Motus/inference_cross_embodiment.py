#!/usr/bin/env python3
"""
Cross-Embodiment Inference Script

Translates a source robot video + action sequence into a target robot video + action sequence.
Uses Wan2.2-Fun-5B-Control as the base video diffusion model with action-conditioned control.

Architecture:
  source_video -> VAE -> latent [B,48,T,H,W]  \
                                                  --> concat --> DiT --> video velocity + action velocity
  source_actions -> ActionControlEncoder [B,48,T,H,W] /

Usage:
  python inference/real_world/Motus/inference_cross_embodiment.py \
    --wan_path /path/to/Wan2.2-Fun-5B-Control \
    --vae_path /path/to/Wan2.2_VAE.pth \
    --source_video /path/to/source.mp4 \
    --source_actions /path/to/actions.npy \
    --output_video output/target.mp4 \
    --output_actions output/target_actions.npy \
    --checkpoint /path/to/checkpoint.pt
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
import imageio

PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cross_embodiment import CrossEmbodimentModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_video_frames(video_path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
        import PIL.Image
        img = PIL.Image.fromarray(frame).convert('RGB').resize((width, height), PIL.Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        frames.append(arr)
    reader.close()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    video = torch.from_numpy(np.stack(frames, axis=0))
    video = video.permute(3, 0, 1, 2).unsqueeze(0)
    video = video * 2.0 - 1.0
    return video


def load_reference_image(image_path: str, height: int, width: int) -> torch.Tensor:
    import PIL.Image
    img = PIL.Image.open(image_path).convert('RGB').resize((width, height), PIL.Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1)
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor


def save_video_frames(video: torch.Tensor, output_path: str, fps: int = 30):
    video = video.squeeze(0).permute(1, 2, 3, 0)
    video = video.cpu().numpy()
    video = (video * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', mode='I')
    for i in range(video.shape[0]):
        writer.append_data(video[i, ...])
    writer.close()
    logger.info(f"Saved video to {output_path}")


def load_actions(action_path: str) -> torch.Tensor:
    actions = np.load(action_path)
    if actions.ndim == 2:
        actions = actions[np.newaxis, ...]
    return torch.from_numpy(actions.astype(np.float32))


def save_actions(actions: torch.Tensor, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    actions_np = actions.squeeze(0).cpu().numpy()
    np.save(output_path, actions_np)
    logger.info(f"Saved actions to {output_path}, shape={actions_np.shape}")


def load_checkpoint(model: CrossEmbodimentModel, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('module', ckpt)

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} keys: {skipped[:5]}...")


def main():
    parser = argparse.ArgumentParser(description='Cross-Embodiment Inference')
    parser.add_argument('--wan_path', type=str, required=True,
                        help='Path to Wan2.2-Fun-5B-Control model directory')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to Wan2.2_VAE.pth')
    parser.add_argument('--source_video', type=str, required=True,
                        help='Path to source robot video (mp4)')
    parser.add_argument('--source_actions', type=str, required=True,
                        help='Path to source actions (.npy), shape [T_a, action_dim]')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Path to output target video (mp4)')
    parser.add_argument('--output_actions', type=str, required=True,
                        help='Path to output target actions (.npy)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned checkpoint (.pt)')
    parser.add_argument('--num_frames', type=int, default=49,
                        help='Number of video frames')
    parser.add_argument('--height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--width', type=int, default=832,
                        help='Video width')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS')
    parser.add_argument('--action_dim', type=int, default=14,
                        help='Action dimension')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--reference_image', type=str, default=None,
                        help='Path to reference image (first frame)')
    parser.add_argument('--t5_checkpoint', type=str, default=None,
                        help='Path to T5 model checkpoint (models_t5_umt5-xxl-enc-bf16.pth)')
    parser.add_argument('--t5_tokenizer', type=str, default=None,
                        help='Path to T5 tokenizer (google/umt5-xxl)')
    parser.add_argument('--clip_checkpoint', type=str, default=None,
                        help='Path to CLIP model checkpoint (safetensors)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = os.path.join(args.wan_path, 'config.json')
    dit_path = os.path.join(args.wan_path, 'diffusion_pytorch_model.safetensors')
    if not os.path.exists(dit_path):
        dit_path = os.path.join(args.wan_path, 'diffusion_pytorch_model.bin')

    logger.info(f"DiT path: {dit_path}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"VAE path: {args.vae_path}")

    model = CrossEmbodimentModel(
        dit_path=dit_path,
        vae_path=args.vae_path,
        config_path=config_path,
        action_dim=args.action_dim,
        num_video_frames=args.num_frames,
        video_height=args.height,
        video_width=args.width,
        fps=args.fps,
        device=args.device,
        precision='bfloat16',
        t5_checkpoint_path=args.t5_checkpoint,
        t5_tokenizer_path=args.t5_tokenizer,
        clip_checkpoint_path=args.clip_checkpoint,
    )

    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)

    model.to(device=args.device, dtype=torch.bfloat16)
    model.eval()

    logger.info(f"Loading source video: {args.source_video}")
    source_video = load_video_frames(args.source_video, args.num_frames, args.height, args.width)
    source_video = source_video.to(device=args.device)

    logger.info(f"Loading source actions: {args.source_actions}")
    source_actions = load_actions(args.source_actions)
    source_actions = source_actions.to(device=args.device)
    logger.info(f"Source actions shape: {source_actions.shape}")

    logger.info("Running inference...")

    context = None
    clip_feature = None
    ref_latents = None

    if args.prompt is not None:
        logger.info(f"Encoding prompt: {args.prompt}")
        context = model.encode_prompt(args.prompt)
        context = context.unsqueeze(0)
        logger.info(f"Context shape: {context.shape}")

    if args.reference_image is not None:
        logger.info(f"Encoding reference image: {args.reference_image}")
        ref_img = load_reference_image(args.reference_image, args.height, args.width)
        ref_img = ref_img.to(device=args.device)
        clip_feature, ref_latents = model.encode_reference_image(ref_img)
        logger.info(f"CLIP feature shape: {clip_feature.shape}, ref_latents shape: {ref_latents.shape}")

    with torch.no_grad():
        target_video, target_actions = model.inference(
            source_video=source_video,
            source_actions=source_actions,
            num_inference_steps=args.num_inference_steps,
            context=context,
            clip_feature=clip_feature,
            ref_latents=ref_latents,
        )

    save_video_frames(target_video, args.output_video, fps=args.fps)
    save_actions(target_actions, args.output_actions)

    logger.info("Inference complete!")


if __name__ == '__main__':
    main()