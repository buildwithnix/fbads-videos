#!/usr/bin/env python3
"""
Simplified version of cluster_processor.py for testing
Removes all async complexity while keeping core functionality
"""

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import requests
import subprocess

import torch
import numpy as np
import cv2
import ray
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.environ.get('CONFIG_FILE', 'config_test.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

VIDEO_URLS = CONFIG['video_urls']
PROCESSING_SETTINGS = CONFIG['processing_settings']
OUTPUT_DIR = Path(PROCESSING_SETTINGS['output_directory'])
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Ray
ray.init(ignore_reinit_error=True)
logger.info("Ray initialized successfully")

def download_video(url, idx):
    """Simple video download"""
    logger.info(f"Downloading video {idx}: {url}")
    try:
        response = requests.get(url, timeout=300)
        temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded video {idx} to {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Failed to download video {idx}: {e}")
        return None

def extract_frames(video_path, max_frames=150):
    """Extract frames from video"""
    logger.info("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames")
    return frames

def create_overlay_text(text, frame_idx):
    """Create simple overlay text"""
    filters = [
        "crop=w=iw:h=ih-100",
        "scale=1080:1920:force_original_aspect_ratio=decrease",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
        f"drawtext=text='{text}':x=(w-text_w)/2:y=1550:fontsize=40:fontcolor=white",
        "drawtext=text='$49 → FREE':x=(w-text_w)/2:y=1650:fontsize=60:fontcolor=red"
    ]
    return ",".join(filters)

def process_video_simple(video_path, output_path, hook_text="LIMITED TIME OFFER"):
    """Simple video processing with FFmpeg"""
    logger.info(f"Processing video with hook: {hook_text}")
    
    # Create FFmpeg filter
    filter_complex = create_overlay_text(hook_text, 0)
    
    # Run FFmpeg
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', filter_complex,
        '-c:v', 'libx264', '-crf', '23',
        '-c:a', 'copy',
        '-y', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Created video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        return False

def main():
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("VIREX-9000 Simplified Test")
    logger.info("=" * 60)
    
    results = []
    
    for idx, video_url in enumerate(VIDEO_URLS):
        logger.info(f"\nProcessing video {idx + 1}/{len(VIDEO_URLS)}")
        
        # Download video
        video_path = download_video(video_url, idx)
        if not video_path:
            continue
        
        # Extract frames for analysis
        frames = extract_frames(video_path)
        logger.info(f"Video has {len(frames)} frames")
        
        # Create variant with overlay
        output_path = OUTPUT_DIR / f"variant_{idx}_0.mp4"
        success = process_video_simple(video_path, str(output_path))
        
        if success:
            results.append({
                "video_idx": idx,
                "output": str(output_path),
                "hook": "LIMITED TIME OFFER",
                "score": 0.85
            })
        
        # Cleanup
        os.unlink(video_path)
    
    # Save results
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Output videos in: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()