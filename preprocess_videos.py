#!/usr/bin/env python3
"""
Preprocess videos into frame folders and optional captions.jsonl entries.

Usage:
  python scripts/preprocess_videos.py --input_dir videos/ --output_dir data/frames --frames_per_clip 8 --stride 4

This extracts frames and writes directory structure:
data/frames/<clip_id>/frame_000.png ... frame_N.png
Optionally you can create a captions.jsonl mapping clip_id -> caption.
"""
import os
import argparse
import shutil
import cv2
import json
from pathlib import Path
from tqdm import tqdm

def extract_clips_from_video(video_path, out_root, frames_per_clip=8, stride=4, caption=None):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
    cap.release()

    clip_index = 0
    for start in range(0, max(1, len(frames) - frames_per_clip + 1), stride):
        clip_frames = frames[start:start + frames_per_clip]
        clip_id = f"{Path(video_path).stem}_clip{clip_index:04d}"
        clip_dir = Path(out_root) / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(clip_frames):
            cv2.imwrite(str(clip_dir / f"frame_{i:03d}.png"), f)
        meta = {"clip_id": clip_id, "source": str(video_path), "start_frame": start, "fps": fps}
        if caption:
            meta["caption"] = caption
        with open(clip_dir / "meta.json", "w") as fh:
            json.dump(meta, fh)
        clip_index += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_per_clip", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--caption", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for video in tqdm(list(input_dir.glob("*.*"))):
        try:
            extract_clips_from_video(video, out_root, frames_per_clip=args.frames_per_clip, stride=args.stride, caption=args.caption)
        except Exception as e:
            print(f"Failed to process {video}: {e}")

if __name__ == "__main__":
    main()
