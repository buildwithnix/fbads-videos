# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a distributed video processing system designed to create viral Facebook ad variants using AI-powered transformations. The system processes turmeric & kojic acid soap advertisement videos, applying neuropsychology-based optimizations to maximize engagement and conversion.

## Core Architecture

### Main Components

1. **cluster_processor.py** - The main processing engine that:
   - Downloads videos from provided URLs
   - Distributes processing across multiple GPUs using Ray
   - Applies AI-based transformations (text removal, style transfer)
   - Optimizes first frames for maximum thumb-stopping power
   - Generates multiple variants per video for A/B testing

2. **config.json** - Central configuration controlling:
   - Video URLs to process
   - Processing settings (batch sizes, quality presets)
   - Model configurations (FLUX, SigLIP, Qwen2-VL)
   - Cluster settings (GPU count)
   - Product-specific optimizations

### Key Technologies

- **Ray** - Distributed computing framework for parallel GPU processing
- **FLUX** - Image-to-image and inpainting pipelines for style transfer
- **SigLIP** - Vision-language model for scene scoring based on viral prompts
- **Qwen2-VL** - OCR model for text detection and removal
- **Grok-4** - LLM for generating viral hooks via X.AI API
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing and frame manipulation

## Development Commands

```bash
# Run the main video processor
python cluster_processor.py

# The system expects a Ray cluster to be initialized
# Videos are downloaded to temp directories and processed results saved to ./results/
```

## Environment Setup

Required environment variables (in .env):
- `XAI_API_KEY` - API key for X.AI Grok model
- `RUNPOD_API_KEY` - RunPod API key (if using RunPod infrastructure)

## High-Level Architecture Flow

1. **Video Download Phase**
   - Concurrent downloads of source videos using aiohttp
   - Videos cached in temp directories

2. **Processing Pipeline**
   - Videos distributed across ModelActor instances (one per GPU)
   - Each variant processes through:
     - Keyframe extraction
     - Text detection and removal (inpainting)
     - Style transfer with product-specific prompts
     - First frame optimization (contrast, golden ratio, glow effects)
     - Temporal smoothing

3. **Optimization Strategies**
   - Distributed caching (DistributedCache actor) for OCR/SigLIP results
   - Batch processing of frames
   - Dynamic batch sizing based on GPU memory
   - Neuropsychology-based prompt engineering for viral appeal

4. **Output Generation**
   - Multiple variants per video (configurable)
   - Results sorted by quality metrics (SSIM scores)
   - Processing summary saved to JSON

## Key Configuration Points

- `variants_per_video`: Number of different versions to create
- `gpu_count`: Number of GPUs in the Ray cluster
- `quality_preset`: Controls inference steps and processing quality
- `first_frame_optimizer`: Special settings for thumb-stopping first frames
- `product_specific.type`: Product type for specialized prompts

## Model Pipeline Details

The system uses a sophisticated multi-model approach:
- FLUX models handle visual transformations
- SigLIP scores frames against viral psychology prompts
- Qwen2-VL detects and helps remove text overlays
- Grok-4 generates dynamic hooks for variants

## Performance Considerations

- GPU memory management through batch sizing
- Distributed caching reduces redundant computations
- Concurrent video downloads minimize I/O wait time
- Ray actor pool manages GPU utilization efficiently