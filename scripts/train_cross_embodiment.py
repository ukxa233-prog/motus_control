#!/usr/bin/env python3
"""
Cross-Embodiment Training Script

Trains a model to translate source robot video + actions into target robot video + actions.
Based on Flow Matching (SFT) training paradigm, similar to DiffSynth-Studio's WanVideo training.

Architecture:
  - Wan2.2-Fun-5B-Control DiT (frozen or LoRA fine-tuned)
  - ActionControlEncoder: source actions -> spatial control features
  - ActionHead: DiT features -> target action predictions

Training data format (JSONL metadata):
  Each line is a JSON object:
  {
    "source_video": "relative/path/to/source.mp4",
    "source_actions": "relative/path/to/source_actions.npy",
    "target_video": "relative/path/to/target.mp4",
    "target_actions": "relative/path/to/target_actions.npy",
    "prompt": "optional text instruction"
  }

Usage:
  python scripts/train_cross_embodiment.py \
    --wan_path pretrained_models/Wan2.2-Fun-5B-Control \
    --vae_path pretrained_models/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth \
    --dataset_metadata_path data/train_metadata.jsonl \
    --dataset_base_path data/ \
    --output_path output/checkpoints \
    --num_frames 49 --height 480 --width 832 \
    --action_dim 14 --fps 30 \
    --batch_size 1 --num_epochs 100 --lr 1e-4
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import imageio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cross_embodiment import CrossEmbodimentModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossEmbodimentDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        base_path: str,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        action_dim: int = 14,
    ):
        self.base_path = Path(base_path)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.action_dim = action_dim

        with open(metadata_path, 'r') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Loaded {len(self.data)} samples from {metadata_path}")

    def __len__(self):
        return len(self.data)

    def _load_video(self, rel_path: str) -> torch.Tensor:
        full_path = self.base_path / rel_path
        reader = imageio.get_reader(str(full_path))
        frames = []
        for i, frame in enumerate(reader):
            if i >= self.num_frames:
                break
            import PIL.Image
            img = PIL.Image.fromarray(frame).convert('RGB').resize(
                (self.width, self.height), PIL.Image.BICUBIC
            )
            arr = np.array(img).astype(np.float32) / 255.0
            frames.append(arr)
        reader.close()

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        video = torch.from_numpy(np.stack(frames, axis=0))
        video = video.permute(3, 0, 1, 2)
        video = video * 2.0 - 1.0
        return video

    def _load_actions(self, rel_path: str) -> torch.Tensor:
        full_path = self.base_path / rel_path
        actions = np.load(str(full_path))
        return torch.from_numpy(actions.astype(np.float32))

    def __getitem__(self, idx):
        item = self.data[idx]

        source_video = self._load_video(item['source_video'])
        source_actions = self._load_actions(item['source_actions'])
        target_video = self._load_video(item['target_video'])
        target_actions = self._load_actions(item['target_actions'])

        prompt = item.get('prompt', '')
        reference_image_path = item.get('reference_image', None)

        return {
            'source_video': source_video,
            'source_actions': source_actions,
            'target_video': target_video,
            'target_actions': target_actions,
            'prompt': prompt,
            'reference_image': reference_image_path,
        }


def collate_fn(batch):
    source_video = torch.stack([item['source_video'] for item in batch], dim=0)
    source_actions = torch.stack([item['source_actions'] for item in batch], dim=0)
    target_video = torch.stack([item['target_video'] for item in batch], dim=0)
    target_actions = torch.stack([item['target_actions'] for item in batch], dim=0)
    prompts = [item['prompt'] for item in batch]
    reference_images = [item['reference_image'] for item in batch]
    return {
        'source_video': source_video,
        'source_actions': source_actions,
        'target_video': target_video,
        'target_actions': target_actions,
        'prompts': prompts,
        'reference_images': reference_images,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        'module': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
    }
    path = os.path.join(output_dir, f'step-{step:08d}.pt')
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint to {path}")
    if is_best:
        best_path = os.path.join(output_dir, 'best.pt')
        torch.save(ckpt, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_reference_image(image_path: str, height: int, width: int) -> torch.Tensor:
    import PIL.Image
    img = PIL.Image.open(image_path).convert('RGB').resize((width, height), PIL.Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1)
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description='Cross-Embodiment Training')
    parser.add_argument('--wan_path', type=str, required=True,
                        help='Path to Wan2.2-Fun-5B-Control model directory')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to Wan2.2_VAE.pth')
    parser.add_argument('--dataset_metadata_path', type=str, required=True,
                        help='Path to training metadata JSONL file')
    parser.add_argument('--dataset_base_path', type=str, required=True,
                        help='Base path for dataset files')
    parser.add_argument('--output_path', type=str, default='output/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--freeze_dit', action='store_true', default=False,
                        help='Freeze DiT weights, only train action encoder/head')
    parser.add_argument('--trainable_models', type=str, default='action_control_encoder,action_head,fps_embedding',
                        help='Comma-separated list of trainable submodules')
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

    trainable_names = [n.strip() for n in args.trainable_models.split(',')]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(tn) for tn in trainable_names)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    model.to(device=args.device, dtype=torch.bfloat16)
    model.train()
    model.wan_model.eval()

    dataset = CrossEmbodimentDataset(
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        action_dim=args.action_dim,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['module'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') and scheduler:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    logger.info(f"Starting training: {args.num_epochs} epochs, {len(dataloader)} batches/epoch")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        epoch_video_loss = 0.0
        epoch_action_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            source_video = batch['source_video'].to(args.device)
            source_actions = batch['source_actions'].to(args.device)
            target_video = batch['target_video'].to(args.device)
            target_actions = batch['target_actions'].to(args.device)

            B = source_video.shape[0]
            timestep = torch.rand(B, device=args.device, dtype=torch.bfloat16)

            context = torch.zeros(B, 1, 4096, device=args.device, dtype=torch.bfloat16)
            clip_feature = None
            ref_latents = None

            prompts = batch.get('prompts', [''] * B)
            reference_images = batch.get('reference_images', [None] * B)

            if prompts[0] and model.text_encoder is not None:
                with torch.no_grad():
                    context_list = model.text_encoder(prompts, args.device)
                    context = torch.stack([
                        torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                        if u.size(0) < model.text_len else u[:model.text_len]
                        for u in context_list
                    ])

            if reference_images[0] is not None and model.image_encoder is not None:
                ref_img = load_reference_image(
                    os.path.join(args.dataset_base_path, reference_images[0]),
                    args.height, args.width
                )
                ref_img = ref_img.to(device=args.device)
                with torch.no_grad():
                    clip_feature, ref_latents = model.encode_reference_image(ref_img)

            loss_dict = model(
                source_video=source_video,
                source_actions=source_actions,
                target_video=target_video,
                target_actions=target_actions,
                timestep=timestep,
                context=context,
                clip_feature=clip_feature,
                ref_latents=ref_latents,
            )

            loss = loss_dict['total_loss'] / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss_dict['total_loss'].item()
            epoch_video_loss += loss_dict['video_loss'].item()
            epoch_action_loss += loss_dict['action_loss'].item()

            if global_step % args.log_interval == 0 and global_step > 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'v_loss': f"{loss_dict['video_loss'].item():.4f}",
                    'a_loss': f"{loss_dict['action_loss'].item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

            if global_step % args.save_interval == 0 and global_step > 0 and (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, args.output_path)

        avg_loss = epoch_loss / len(dataloader)
        avg_v_loss = epoch_video_loss / len(dataloader)
        avg_a_loss = epoch_action_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, v_loss={avg_v_loss:.4f}, a_loss={avg_a_loss:.4f}")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_path = os.path.join(args.output_path, 'best.pt')
            os.makedirs(args.output_path, exist_ok=True)
            torch.save({
                'module': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch + 1,
                'step': global_step,
            }, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()