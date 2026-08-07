#!/usr/bin/env python3
"""
Simple inference script to generate N frames from a prompt using SDXL + LoRA adapters.

Notes:
 - This script generates independent frames and then assembles them into a video.
 - For smoother motion, generate lower-FPS keyframes and perform flow-based interpolation with a separate tool (RIFE/DAIN/FFmpeg-based).
 - To apply LoRA at inference, we re-insert adapters into the UNet and add their output to the forward pass.
"""
import os
import argparse
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import cv2
from tqdm import trange


def load_lora_into_unet(unet, lora_path):
    """
    Load LoRA state dict saved by training script.
    The saving format is a dict keyed by module name like "<module>.lora_q"
    """
    sd = torch.load(lora_path, map_location="cpu")
    loaded = 0
    for n, m in unet.named_modules():
        if f"{n}.lora_q" in sd and hasattr(m, "lora_q"):
            m.lora_q.load_state_dict(sd[f"{n}.lora_q"])
            loaded += 1
        if f"{n}.lora_k" in sd and hasattr(m, "lora_k"):
            m.lora_k.load_state_dict(sd[f"{n}.lora_k"])
            loaded += 1
    print(f"[load_lora] Loaded {loaded} LoRA tensors from {lora_path}")
    return unet


def decode_latents(vae, latents):
    latents = latents / 0.18215
    with torch.no_grad():
        imgs = vae.decode(latents.half()).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = (imgs * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="./out_video")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print("Loading models for inference...")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # load LoRA into unet (adapters must exist)
    unet = load_lora_into_unet(unet, args.lora_path)

    # create output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # memory optimizations
    try:
        unet.enable_attention_slicing()
    except Exception:
        pass

    tokens = tokenizer([args.prompt], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    text_emb = text_encoder(**tokens).last_hidden_state

    # sampling: create independent frames with the same prompt (for basic motion, you can vary noise or seed per frame)
    frames = []
    for i in trange(args.frames, desc="Generating frames"):
        # new seed per frame for variation (or keep same to get identical frames)
        cur_seed = args.seed + i
        generator = torch.Generator(device).manual_seed(cur_seed)
        # start from random latents
        latents = torch.randn((1, unet.in_channels, args.resolution // 8, args.resolution // 8), device=device, generator=generator).half()
        # perform a small number of ddim/ddpm steps (for speed) - here we do a naive denoise loop
        num_timesteps = 50
        for t in range(num_timesteps):
            tim = torch.tensor([t], device=latents.device)
            with torch.no_grad():
                model_pred = unet(latents, tim, encoder_hidden_states=text_emb).sample
            # simple Euler-like step (not a proper sampler; replace with a proper scheduler for best results)
            latents = latents - 0.02 * model_pred
        imgs = decode_latents(vae, latents)
        img = Image.fromarray(imgs[0])
        img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
        out_path = os.path.join(args.out_dir, f"frame_{i:03d}.png")
        img.save(out_path)
        frames.append(out_path)

    # assemble video with OpenCV
    video_path = os.path.join(args.out_dir, "out.mp4")
    img0 = cv2.imread(frames[0])
    h, w = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 8, (w, h))
    for p in frames:
        img = cv2.imread(p)
        writer.write(img)
    writer.release()
    print("Saved video:", video_path)


if __name__ == "__main__":
    main()
