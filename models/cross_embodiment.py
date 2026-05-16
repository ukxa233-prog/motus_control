import sys
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

DIFFSYNTH_ROOT = "/home/a100/youjingzhou/codes/DiffSynth-Studio"
if DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, DIFFSYNTH_ROOT)

from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from diffsynth.models.wan_video_image_encoder import WanImageEncoder

logger = logging.getLogger(__name__)


class ActionControlEncoder(nn.Module):
    def __init__(self, action_dim: int = 14, hidden_dim: int = 256, out_channels: int = 48):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, actions: torch.Tensor, target_T: int, target_H: int, target_W: int) -> torch.Tensor:
        B, T_a, _ = actions.shape

        x = self.action_proj(actions)
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_T, mode='linear', align_corners=False)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1, -1, -1, target_H, target_W)

        x = self.spatial_conv(x)

        return x


class ActionHead(nn.Module):
    def __init__(self, feature_dim: int = 3072, hidden_dim: int = 512, action_dim: int = 14):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor, grid_sizes: torch.Tensor, target_T_a: int, video_start: int = 0) -> torch.Tensor:
        B = features.shape[0]
        T_patch, H_patch, W_patch = grid_sizes[0].tolist()
        f = features[:, video_start:video_start + T_patch * H_patch * W_patch, :]
        f = f.reshape(B, T_patch, H_patch, W_patch, self.feature_dim)
        f = f.mean(dim=(2, 3))
        f = self.norm(f)
        actions = self.proj(f)
        if actions.shape[1] != target_T_a:
            actions = actions.transpose(1, 2)
            actions = F.interpolate(actions, size=target_T_a, mode='linear', align_corners=False)
            actions = actions.transpose(1, 2)
        return actions


class CrossEmbodimentModel(nn.Module):
    def __init__(
        self,
        dit_path: str,
        vae_path: str,
        config_path: str,
        action_dim: int = 14,
        num_video_frames: int = 49,
        video_height: int = 480,
        video_width: int = 832,
        fps: int = 30,
        device: str = "cuda",
        precision: str = "bfloat16",
        t5_checkpoint_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
        clip_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_video_frames = num_video_frames
        self.video_height = video_height
        self.video_width = video_width
        self.fps = fps

        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

        with open(config_path, 'r') as f:
            model_config = json.load(f)

        wan_valid_keys = {
            'model_type', 'patch_size', 'text_len', 'in_dim', 'dim', 'ffn_dim',
            'freq_dim', 'text_dim', 'out_dim', 'num_heads', 'num_layers',
            'window_size', 'qk_norm', 'cross_attn_norm', 'eps'
        }
        wan_config = {k: v for k, v in model_config.items() if k in wan_valid_keys}
        self.wan_model = WanModel(**wan_config)
        self._load_dit_weights(dit_path)
        self.wan_model.to(device=self.device, dtype=self.dtype)
        self.wan_model.eval()

        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device, dtype=self.dtype)

        self.in_dim = model_config.get('in_dim', 148)
        self.dim = model_config.get('dim', 3072)
        self.num_heads = model_config.get('num_heads', 24)
        self.num_layers = model_config.get('num_layers', 30)
        self.freq_dim = model_config.get('freq_dim', 256)
        self.text_len = model_config.get('text_len', 512)

        self.action_control_encoder = ActionControlEncoder(
            action_dim=action_dim, hidden_dim=256, out_channels=48
        )

        self.action_head = ActionHead(
            feature_dim=self.dim, hidden_dim=512, action_dim=action_dim
        )

        self._setup_fps_embedding()

        self.latent_T = (num_video_frames - 1) // 4 + 1
        self.latent_H = video_height // 16
        self.latent_W = video_width // 16

        self._init_text_encoder(t5_checkpoint_path, t5_tokenizer_path)
        self._init_image_encoder(clip_checkpoint_path)
        self._init_img_emb_and_ref_conv()

        logger.info(f"CrossEmbodimentModel: latent_T={self.latent_T}, latent_H={self.latent_H}, latent_W={self.latent_W}")
        logger.info(f"  in_dim={self.in_dim}, dim={self.dim}, num_layers={self.num_layers}")

    def _init_text_encoder(self, t5_checkpoint_path: Optional[str], t5_tokenizer_path: Optional[str]):
        if t5_checkpoint_path is not None and t5_tokenizer_path is not None:
            self.text_encoder = T5EncoderModel(
                text_len=self.text_len,
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=t5_checkpoint_path,
                tokenizer_path=t5_tokenizer_path,
            )
            logger.info(f"T5 text encoder loaded from {t5_checkpoint_path}")
        else:
            self.text_encoder = None
            logger.info("T5 text encoder not configured")

    def _init_image_encoder(self, clip_checkpoint_path: Optional[str]):
        if clip_checkpoint_path is not None and os.path.exists(clip_checkpoint_path):
            self.image_encoder = WanImageEncoder()
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(clip_checkpoint_path)
            state_dict_ = {}
            for name in state_dict:
                if name.startswith('textual.'):
                    continue
                state_dict_['model.' + name] = state_dict[name]
            self.image_encoder.load_state_dict(state_dict_, strict=False)
            self.image_encoder.to(device=self.device, dtype=self.dtype)
            self.image_encoder.eval()
            logger.info(f"CLIP image encoder loaded from {clip_checkpoint_path}")
        else:
            self.image_encoder = None
            logger.info("CLIP image encoder not configured")

    def _init_img_emb_and_ref_conv(self):
        self.img_emb = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.ref_conv = nn.Conv2d(48, self.dim, kernel_size=(2, 2), stride=(2, 2))
        self.img_emb.to(device=self.device, dtype=self.dtype)
        self.ref_conv.to(device=self.device, dtype=self.dtype)
        logger.info("img_emb and ref_conv initialized")

    def _load_dit_weights(self, dit_path: str):
        if dit_path.endswith('.safetensors'):
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(dit_path, device='cpu')
        else:
            state_dict = torch.load(dit_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if any(k.startswith('dit.') for k in state_dict.keys()):
            state_dict = {k[4:]: v for k, v in state_dict.items()}
        missing, unexpected = self.wan_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing DiT keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected DiT keys: {unexpected}")
        logger.info(f"Loaded DiT weights from {dit_path}")

    def _setup_fps_embedding(self):
        self.fps_embed_dim = 4
        self.fps_embedding = nn.Parameter(
            torch.randn(1, self.fps_embed_dim, 1, 1, 1) * 0.02
        )

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.encode(video.to(self.dtype))

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frames = []
            for i in range(latent.shape[0]):
                pixels = self.vae.decode([latent[i]])[0]
                frames.append(pixels)
            return torch.stack(frames, dim=0).to(self.dtype)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("T5 text encoder not configured. Provide t5_checkpoint_path and t5_tokenizer_path.")
        with torch.no_grad():
            context_list = self.text_encoder([prompt], self.device)
            context = context_list[0]
            if context.shape[0] < self.text_len:
                pad = torch.zeros(self.text_len - context.shape[0], context.shape[1],
                                  device=self.device, dtype=self.dtype)
                context = torch.cat([context, pad], dim=0)
            elif context.shape[0] > self.text_len:
                context = context[:self.text_len]
            return context

    def encode_reference_image(self, reference_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.image_encoder is None:
            raise RuntimeError("CLIP image encoder not configured. Provide clip_checkpoint_path.")
        with torch.no_grad():
            ref_img_input = reference_image.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            clip_feature = self.image_encoder.encode_image([ref_img_input])
            clip_feature = clip_feature.to(self.dtype)

            ref_video = reference_image.unsqueeze(0).unsqueeze(2)
            ref_video = ref_video.to(self.dtype)
            ref_latents = self.vae.encode(ref_video)
            ref_latents = ref_latents.squeeze(2)

        return clip_feature, ref_latents

    def _prepare_dit_input(
        self,
        noisy_latent: torch.Tensor,
        source_video_latent: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]
        fps_embed = self.fps_embedding.expand(B, -1, self.latent_T, self.latent_H, self.latent_W).to(
            device=self.device, dtype=self.dtype
        )
        y = torch.cat([source_video_latent, action_features, fps_embed], dim=1)
        x = torch.cat([noisy_latent, y], dim=1)
        return x

    def _dit_forward_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.wan_model.patch_embedding.weight.device
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)

        B = x.shape[0]
        x_list = [x[i] for i in range(B)]
        x_patched = [self.wan_model.patch_embedding(u.unsqueeze(0)) for u in x_list]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x_patched])
        x_flat = [u.flatten(2).transpose(1, 2) for u in x_patched]
        seq_lens = torch.tensor([u.size(1) for u in x_flat], dtype=torch.long, device=device)
        seq_len = seq_lens.max().item()
        x_tokens = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x_flat
        ])

        if ref_latents is not None:
            ref_tokens = self.ref_conv(ref_latents)
            _, _, H_ref, W_ref = ref_tokens.shape
            ref_tokens = ref_tokens.flatten(2).transpose(1, 2)
            ref_len = ref_tokens.shape[1]
            x_tokens = torch.cat([ref_tokens, x_tokens], dim=1)
            seq_lens = seq_lens + ref_len
            seq_len = x_tokens.shape[1]

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).bfloat16()
            )
            e0 = self.wan_model.time_projection(e).unflatten(2, (6, self.wan_model.dim))

        context = self.wan_model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.wan_model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_feature is not None:
            clip_emb = self.img_emb(clip_feature)
            context = torch.cat([clip_emb, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.wan_model.freqs,
            context=context,
            context_lens=None,
        )

        for block in self.wan_model.blocks:
            x_tokens = block(x_tokens, **kwargs)

        return x_tokens, e, grid_sizes

    def forward(
        self,
        source_video: torch.Tensor,
        source_actions: torch.Tensor,
        target_video: torch.Tensor,
        target_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
    ) -> dict:
        B = source_video.shape[0]

        source_video = source_video.to(self.dtype)
        source_actions = source_actions.to(self.dtype)
        target_video = target_video.to(self.dtype)
        target_actions = target_actions.to(self.dtype)

        with torch.no_grad():
            source_latent = self.encode_video(source_video)
            target_latent = self.encode_video(target_video)

        action_features = self.action_control_encoder(
            source_actions, self.latent_T, self.latent_H, self.latent_W
        )

        sigma = timestep.view(B, 1, 1, 1, 1)
        noise = torch.randn_like(target_latent, dtype=self.dtype)
        noisy_latent = target_latent * (1 - sigma) + noise * sigma
        video_target = noise - target_latent

        x = self._prepare_dit_input(noisy_latent, source_latent, action_features)

        t_scaled = (timestep * 1000).to(self.dtype)
        features, e, grid_sizes = self._dit_forward_features(
            x, t_scaled, context, clip_feature=clip_feature, ref_latents=ref_latents
        )

        video_velocity = self.wan_model.head(features, e)
        video_velocity = self.wan_model.unpatchify(video_velocity, grid_sizes)
        video_velocity = torch.stack([u for u in video_velocity], dim=0)

        video_start = 0
        if ref_latents is not None:
            _, _, H_in, W_in = ref_latents.shape
            H_out = (H_in - 2) // 2 + 1
            W_out = (W_in - 2) // 2 + 1
            video_start = H_out * W_out

        action_pred = self.action_head(features, grid_sizes, source_actions.shape[1], video_start=video_start)

        video_loss = F.mse_loss(video_velocity.float(), video_target.float(), reduction='mean')
        action_loss = F.mse_loss(action_pred.float(), target_actions.float(), reduction='mean')
        total_loss = video_loss + action_loss

        return {
            'total_loss': total_loss,
            'video_loss': video_loss,
            'action_loss': action_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        source_video: torch.Tensor,
        source_actions: torch.Tensor,
        num_inference_steps: int = 50,
        context: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = source_video.shape[0]

        source_video = source_video.to(self.dtype)
        source_actions = source_actions.to(self.dtype)

        source_latent = self.encode_video(source_video)

        action_features = self.action_control_encoder(
            source_actions, self.latent_T, self.latent_H, self.latent_W
        )

        if context is None:
            context = torch.zeros(B, 1, 4096, device=self.device, dtype=self.dtype)

        latent = torch.randn(
            (B, 48, self.latent_T, self.latent_H, self.latent_W),
            device=self.device, dtype=self.dtype
        )

        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)

        for i in range(num_inference_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            x = self._prepare_dit_input(latent, source_latent, action_features)

            t_scaled = (t * 1000).expand(B).to(self.dtype)
            features, e, grid_sizes = self._dit_forward_features(
                x, t_scaled, context, clip_feature=clip_feature, ref_latents=ref_latents
            )

            video_velocity = self.wan_model.head(features, e)
            video_velocity = self.wan_model.unpatchify(video_velocity, grid_sizes)
            video_velocity = torch.stack([u for u in video_velocity], dim=0)

            latent = latent + video_velocity * dt

        target_video = self.decode_video(latent)
        target_video = (target_video + 1.0) / 2.0
        target_video = torch.clamp(target_video, 0, 1)

        x_final = self._prepare_dit_input(latent, source_latent, action_features)
        t_zero = torch.zeros(B, device=self.device, dtype=self.dtype)
        features_final, _, grid_sizes_final = self._dit_forward_features(
            x_final, t_zero, context, clip_feature=clip_feature, ref_latents=ref_latents
        )

        video_start = 0
        if ref_latents is not None:
            _, _, H_in, W_in = ref_latents.shape
            H_out = (H_in - 2) // 2 + 1
            W_out = (W_in - 2) // 2 + 1
            video_start = H_out * W_out

        target_actions = self.action_head(
            features_final, grid_sizes_final, source_actions.shape[1], video_start=video_start
        )

        return target_video.float(), target_actions.float()