#!/usr/bin/env python3
"""
Low-VRAM LoRA training script optimized for RTX 4060 and SDXL.

Key strategies included:
 - Train only small LoRA adapters (few MB)
 - Use bitsandbytes 8-bit AdamW optimizer
 - Mixed precision (fp16)
 - Gradient accumulation to simulate larger batches
 - Gradient checkpointing on UNet
 - Attention slicing and xformers if available
 - Small resolution (512) recommended for 4060

Notes about SDXL:
 - SDXL architecture differs from SDv1.5. Depending on the SDXL checkpoint you use, there may be two text encoders (text_encoder and text_encoder_2) and multiple U-Nets (base + refiner).
 - This script uses heuristics to attach LoRA adapters to attention modules. For perfect compatibility with a specific SDXL variant, you may need to adjust adapter placement.

Usage:
 accelerate launch scripts/train_lora_4060_sdxl.py --model_id stabilityai/stable-diffusion-xl-base-1.0 --dataset_dir ./data/frames --output_dir ./lora_out --resolution 512
"""
import os
import argparse
import math
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb
from PIL import Image
import glob
import random
from tqdm import tqdm

# ---------- LoRA Adapter ----------
class LoRAAdapter(torch.nn.Module):
    def __init__(self, orig_dim, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = torch.nn.Linear(orig_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, orig_dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        return self.up(self.down(x)) * (self.alpha / self.rank)


def attach_lora_to_unet(unet, lora_rank=4):
    """
    Attach LoRA adapters to candidate modules in UNet.
    Heuristic: find modules with attributes to_q/to_k/to_v (common in diffusers attention)
    """
    attached = 0
    for name, module in unet.named_modules():
        # Heuristic detection
        if hasattr(module, "to_q") or hasattr(module, "to_k") or hasattr(module, "to_v"):
            if hasattr(module, "to_q"):
                try:
                    qdim = module.to_q.out_features if hasattr(module.to_q, "out_features") else module.to_q.weight.shape[0]
                except Exception:
                    qdim = None
                if qdim:
                    module.lora_q = LoRAAdapter(qdim, rank=lora_rank)
            if hasattr(module, "to_k"):
                try:
                    kdim = module.to_k.out_features if hasattr(module.to_k, "out_features") else module.to_k.weight.shape[0]
                except Exception:
                    kdim = None
                if kdim:
                    module.lora_k = LoRAAdapter(kdim, rank=lora_rank)
            attached += 1
    print(f"[attach_lora] Attached LoRA to {attached} attention modules (heuristic).")
    return unet

# ---------- Dataset ----------
class FramesDataset(Dataset):
    def __init__(self, root_dir, resolution=512):
        # expects structure: root_dir/<clip_id>/frame_000.png ... and meta.json optional
        self.resolution = resolution
        self.samples = []
        for clip_dir in glob.glob(os.path.join(root_dir, "*")):
            if os.path.isdir(clip_dir):
                frames = sorted(glob.glob(os.path.join(clip_dir, "frame_*.png")))
                if len(frames) == 0:
                    continue
                # optional caption
                meta_path = os.path.join(clip_dir, "meta.json")
                caption = None
                if os.path.exists(meta_path):
                    try:
                        import json
                        meta = json.load(open(meta_path))
                        caption = meta.get("caption")
                    except Exception:
                        caption = None
                self.samples.append({"clip": clip_dir, "frames": frames, "caption": caption})
        # fallback: if root dir contains individual images, add them as single-frame clips
        for img in glob.glob(os.path.join(root_dir, "*.png")) + glob.glob(os.path.join(root_dir, "*.jpg")):
            self.samples.append({"clip": img, "frames": [img], "caption": None})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # choose a random frame from the clip to train LoRA for appearance/style
        frame_path = random.choice(sample["frames"])
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        pixel_values = torch.tensor(( (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))).float().view(self.resolution, self.resolution, 3).permute(2,0,1) / 255.0 ))
        caption = sample.get("caption") or "A cinematic short clip"
        return {"pixel_values": pixel_values, "caption": caption}

# ---------- Training ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_out")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--use_xformers", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    # SDXL compatibility: adaptively try to load common components
    try:
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    except Exception as e:
        print("Failed to load UNet with subfolder 'unet' - trying default load:", e)
        unet = UNet2DConditionModel.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)

    try:
        vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
    except Exception:
        vae = AutoencoderKL.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)

    # Tokenizers / text encoders: SDXL often uses two text encoders; attempt best-effort
    try:
        tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    except Exception:
        # fallback to a generic text encoder tokenizer
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Freeze base weights
    for p in unet.parameters(): p.requires_grad = False
    for p in vae.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False

    # Patch LoRA into UNet
    print("Attaching LoRA adapters...")
    unet = attach_lora_to_unet(unet, lora_rank=args.lora_rank)

    # Collect LoRA parameters (trainable)
    lora_params = []
    for n, p in unet.named_parameters():
        if "lora" in n:
            p.requires_grad = True
            lora_params.append(p)
    total_trainable = sum(p.numel() for p in lora_params)
    print(f"Total trainable LoRA params: {total_trainable:,}")

    # Optimizer (8-bit)
    optimizer = bnb.optim.AdamW8bit(lora_params, lr=args.lr)

    # Scheduler (basic)
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # Accelerator
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.grad_accum_steps, cpu=False)
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Memory optimizations
    try:
        unet.enable_attention_slicing()
    except Exception:
        pass
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Gradient checkpointing (reduces memory, increases time)
    try:
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
    except Exception:
        pass

    dataset = FramesDataset(args.dataset_dir, resolution=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print("Starting training loop")
    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # tokenize captions
                texts = batch["caption"]
                tokens = tokenizer(list(texts), padding="max_length", truncation=True, max_length=77, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                text_emb = text_encoder(**tokens).last_hidden_state

                # encode images to latents
                pixel_values = batch["pixel_values"].to(device).half()
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # noise & timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # predict
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

                loss = torch.nn.functional.mse_loss(model_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % 50 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f} global_step {global_step}")

        # Save LoRA weights only at epoch end
        lora_state = {}
        for n, m in unet.named_modules():
            if hasattr(m, "lora_q"):
                lora_state[f"{n}.lora_q"] = m.lora_q.state_dict()
            if hasattr(m, "lora_k"):
                lora_state[f"{n}.lora_k"] = m.lora_k.state_dict()
        out_path = os.path.join(args.output_dir, f"lora_epoch{epoch}.pt")
        torch.save(lora_state, out_path)
        print(f"Saved LoRA epoch {epoch} -> {out_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "lora_final.pt")
    torch.save(lora_state, final_path)
    print("Training complete. Final LoRA saved to:", final_path)


if __name__ == "__main__":
    main()
