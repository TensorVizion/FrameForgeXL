Text→Video pipeline + LoRA (RTX 4060 / low-VRAM)
This repository contains a low-VRAM-focused starter pipeline to:

Fine-tune LoRA adapters for a Stable-Diffusion-style U-Net on video frames (or frame datasets).
Inference pipeline to render short clips from text prompts and apply simple post-processing.
Practical configuration tuned for an RTX 4060 (≈8GB VRAM).
Quick summary

Target GPU: RTX 4060 (8GB). All scripts aim to keep VRAM usage minimal: batch_size=1, gradient_accumulation, 8-bit optimizer, mixed precision (fp16), gradient checkpointing, attention slicing, optional CPU offload.
Base model: any diffusers-compatible stable-diffusion checkpoint (default: runwayml/stable-diffusion-v1-5). For temporal-aware models (AnimateDiff-style), adaption is noted in comments.
LoRA style: small adapters applied to attention projection matrices. Only LoRA weights are trained and saved.
Contents

requirements.txt — Python packages to install.
scripts/train_lora_4060.py — LoRA training script optimized for low VRAM.
scripts/infer_video_4060.py — Inference script to generate frames from prompts and assemble a short video.
scripts/preprocess_videos.py — Helper to extract frames from video files and prepare dataset shards.
Important notes

The starter LoRA focuses on spatial/style adapters (per-frame). Extending to temporal LoRA requires a temporal-attention-capable U-Net (AnimateDiff/Tune-A-Video style). I include notes inside the training script how to adapt to temporal attention modules if you use such a U-Net.
Expect to work at 512×512 or lower for training on a 4060. For higher resolution, use lower batch sizes and stronger offload (or use keyframe+interpolation strategies).
If you want me to push these into a GitHub repo or adapt to a specific base checkpoint (e.g., an AnimateDiff checkpoint), tell me and I will prepare that.
Quick start (recommended)

Create a Python virtualenv and install dependencies:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Prepare dataset:

Use scripts/preprocess_videos.py to extract frames and captions (or point training script at a folder of frames + captions.jsonl).
Train LoRA:

Use the accelerate launcher: accelerate launch scripts/train_lora_4060.py --model_id runwayml/stable-diffusion-v1-5 --dataset_dir ./data/frames --output_dir ./lora_out --resolution 512 --epochs 3
See training script args for more tuning.
Inference (generate a short clip):

python scripts/infer_video_4060.py --model_id runwayml/stable-diffusion-v1-5 --lora_path ./lora_out/lora_final.pt --prompt "A neon cyberpunk car driving through rainy night" --frames 8 --resolution 512
Helpful environment vars (reduce fragmentation)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
If you want, I can:

Convert this scaffold into a GitHub repo and push it.
Customize the LoRA adapter placement to an AnimateDiff checkpoint for temporal LoRA training.
Add a minimal Colab notebook for interactive runs (not ideal for 4060 but useful for sharing).
Read the script docstrings for details and next steps.
