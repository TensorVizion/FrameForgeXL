# FrameForgeXL - SDXL Text→Video Pipeline (Low-VRAM, RTX 4060)

This repository addition provides a low-VRAM-focused starter pipeline for text→video using SDXL as the base model, with LoRA adapters designed to be trainable on an RTX 4060 (≈8GB VRAM).

Overview
- Target GPU: RTX 4060 (8GB). All scripts favor memory-sparing techniques: batch_size=1, gradient accumulation, bitsandbytes 8-bit optimizer, mixed precision (fp16), gradient checkpointing, attention slicing, optional CPU offload.
- Base model: SDXL (stabilityai/stable-diffusion-xl-base-1.0) by default. SDXL has a different architecture vs SDv1.x; the scripts include heuristics to find attention modules and attach LoRA adapters accordingly and include notes where manual adjustments may be needed.
- LoRA style: multiple adapters inserted where possible:
  - Attention projection LoRAs for query/key/value (UNet attention blocks)
  - Optional text-encoder LoRA (for CLIP-like encoders)
  - Hooks for temporal LoRA adapters (if you use an AnimateDiff/Tune-A-Video style UNet with temporal attention)

Files added
- scripts/preprocess_videos.py — extract clips and frames; create per-clip metadata
- scripts/train_lora_4060_sdxl.py — LoRA training script optimized for low VRAM and SDXL
- scripts/infer_video_4060_sdxl.py — inference script to generate short clips using SDXL + LoRA
- requirements-sdxl.txt — dependency pins (SDXL-friendly)
- .gitignore — standard ignores

Important notes
- SDXL is large and may still be challenging on an 8GB GPU. These scripts emphasize LoRA-only training and many memory optimizations, but you may need to use CPU offload or further reduce resolution (512) for training.
- If you plan to use SDXL refiner and base pairs, or a specific community SDXL checkpoint, adjust the --model_id argument accordingly.

Quick start
1. Create venv and install deps:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-sdxl.txt

2. Prepare clip frames:
   python scripts/preprocess_videos.py --input_dir videos/ --output_dir data/frames --frames_per_clip 8 --stride 4

3. Configure accelerate (single GPU, mixed precision=fp16):
   accelerate config

4. Train LoRA (example):
   accelerate launch scripts/train_lora_4060_sdxl.py --model_id stabilityai/stable-diffusion-xl-base-1.0 --dataset_dir ./data/frames --output_dir ./lora_out --resolution 512 --epochs 3 --batch_size 1 --grad_accum_steps 4

5. Inference:
   python scripts/infer_video_4060_sdxl.py --model_id stabilityai/stable-diffusion-xl-base-1.0 --lora_path ./lora_out/lora_final.pt --prompt "A neon cyberpunk car driving through rainy night" --frames 8 --resolution 512 --out_dir ./out

If you'd like, I can:
- Modify LoRA placement specifically to your SDXL checkpoint if you provide the exact ID
- Push adjustments to use SDXL base+refiner pair
- Add a GitHub Actions workflow to run lightweight lint/tests

---
