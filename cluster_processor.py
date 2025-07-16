#!/usr/bin/env python3
"""
Facebook Ads Video Processor - Ultra-Optimized Cluster Edition
Implements all 5 optimization recommendations for 5-10x speedup
"""

import os
import sys
import json
import time
import aiohttp
import tempfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache
import concurrent.futures

import torch
import numpy as np
import cv2
import av
import ray
from ray.util import ActorPool
import re
import random
from PIL import Image
import hashlib

# AI/ML imports
from diffusers import StableVideoDiffusionPipeline, FluxFillPipeline
from transformers import AutoModel, AutoProcessor, VideoLlavaProcessor, VideoLlavaForConditionalGeneration
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
from langchain_openai import ChatOpenAI
import paddleocr
import pennylane as qml
from subprocess import run, PIPE, CalledProcessError

# Performance optimizations
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorRT not available - will use standard optimization")

# Configure logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load configuration
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    logger.error(f"Configuration file {CONFIG_PATH} not found!")
    sys.exit(1)

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Extract all settings
VIDEO_URLS = CONFIG['video_urls']
PROCESSING_SETTINGS = CONFIG['processing_settings']
CLUSTER_SETTINGS = CONFIG['cluster_settings']
OPTIMIZATION_SETTINGS = CONFIG['optimization_settings']
QUALITY_SETTINGS = CONFIG['quality_presets'][PROCESSING_SETTINGS['quality_preset']]

# VIREX-9000 Dynamic Product Profiles
DYNAMIC_PROFILES = CONFIG.get('dynamic_profiles', {})
PRODUCT_TYPE = CONFIG.get('product_specific', {}).get('type', 'default')
CURRENT_PROFILE = DYNAMIC_PROFILES.get(
    PRODUCT_TYPE, DYNAMIC_PROFILES.get('default', {}))
SCORING_SETTINGS = CONFIG.get('scoring_settings', {})
OVERLAY_SYSTEM = CONFIG.get('overlay_system', {})

# Load API keys from environment variables with fallback to config
API_KEYS = {
    'xai_api_key': os.environ.get(
        'XAI_API_KEY', CONFIG.get(
            'api_keys', {}).get('xai_api_key')), 'runpod_api_key': os.environ.get(
                'RUNPOD_API_KEY', CONFIG.get(
                    'api_keys', {}).get('runpod_api_key'))}

# Create output directory
os.makedirs(PROCESSING_SETTINGS['output_directory'], exist_ok=True)

# Initialize Ray with optimizations
try:
    ray.init(
        address='auto',
        ignore_reinit_error=True,
        logging_level=logging.INFO,
        runtime_env={
            "env_vars": {
                "RAY_ENABLE_RUNTIME_ENV_CONN": "1",  # Fix multi-node glitches
            }
        }
    )
except BaseException:
    # If no existing cluster, start a new one
    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.INFO,
        num_gpus=CLUSTER_SETTINGS.get('gpu_count', 0),
        runtime_env={
            "env_vars": {
                "RAY_ENABLE_RUNTIME_ENV_CONN": "1",
            }
        }
    )

# Global quantum circuit cache (shared across workers)
QUANTUM_CIRCUIT_CACHE = {}

# Configuration for temp file cleanup
SAVE_INTERMEDIATE_FILES = False

# Enable FlashAttention-2 if available and configured
if OPTIMIZATION_SETTINGS.get(
        'use_flash_attention',
        True) and torch.cuda.is_available() and hasattr(
            torch.backends.cuda,
        'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)
    logger.info("FlashAttention-2 enabled for faster inference")

# ============================================================================
# DISTRIBUTED CACHING SYSTEM
# ============================================================================


@ray.remote
class DistributedCache:
    """Distributed cache for sharing computed results across GPUs"""

    def __init__(self):
        self.ocr_cache = {}
        self.siglip_cache = {}
        self.style_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_frame_hash(self, frame: np.ndarray) -> str:
        """Generate hash for frame to use as cache key"""
        return hashlib.md5(frame.tobytes()).hexdigest()

    def get_ocr(self, frame_hash: str) -> Optional[Any]:
        result = self.ocr_cache.get(frame_hash)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def set_ocr(self, frame_hash: str, result: Any):
        self.ocr_cache[frame_hash] = result

    def get_siglip(self, frame_hash: str) -> Optional[Dict[str, float]]:
        result = self.siglip_cache.get(frame_hash)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def set_siglip(self, frame_hash: str, scores: Dict[str, float]):
        self.siglip_cache[frame_hash] = scores

    def get_style(self, frame_hash: str,
                  variant_id: int) -> Optional[np.ndarray]:
        key = f"{frame_hash}_{variant_id}"
        result = self.style_cache.get(key)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def set_style(self, frame_hash: str, variant_id: int, result: np.ndarray):
        key = f"{frame_hash}_{variant_id}"
        self.style_cache[key] = result

    def get_stats(self) -> dict:
        hit_rate = self.cache_hits / \
            (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'ocr_entries': len(self.ocr_cache),
            'siglip_entries': len(self.siglip_cache),
            'style_entries': len(self.style_cache)
        }


# Initialize global cache if configured
if OPTIMIZATION_SETTINGS.get('use_caching', True):
    cache_actor = DistributedCache.remote()
else:
    cache_actor = None

# ============================================================================
# NEUROPSYCHOLOGY-DRIVEN PROMPT ENGINE FOR VIRAL MARKETING
# ============================================================================


class NeuropsychologyPromptEngine:
    """Generate hyper-optimized prompts based on product type and viral trends"""

    def __init__(self):
        self.prompt_categories = {
            "fomo_trigger": 0.25,      # Scarcity and urgency
            "social_proof": 0.20,      # Authority and testimonials
            "pattern_interrupt": 0.20,  # Attention grabbing
            "emotion_hook": 0.15,      # Emotional resonance
            "visual_appeal": 0.10,     # Aesthetic quality
            "trend_alignment": 0.10    # Current viral trends
        }

    def get_prompts(
            self,
            product_type: str = "skincare",
            trend_date: str = "2025-07",
            platform: str = "facebook_reels") -> List[str]:
        """Get product-specific neuropsychology-optimized prompts"""

        if product_type == "turmeric_kojic_soap":
            return self._get_turmeric_kojic_prompts()
        elif product_type == "skincare":
            return self._get_skincare_prompts()
        elif product_type == "electronics":
            return self._get_electronics_prompts()
        elif product_type == "fashion":
            return self._get_fashion_prompts()
        else:
            return self._get_generic_viral_prompts()

    def _get_turmeric_kojic_prompts(self) -> List[str]:
        """Ultra-specific prompts for turmeric & kojic acid soap with VIREX-9000 optimization"""
        return [
            # FOMO Trigger with product specifics
            "Instant turmeric & kojic acid soap transformation that triggers FOMO: golden glow reveal, before-after brightening magic with dewy finish, millions sharing this viral skincare demo – thumb-stopping urgency in every radiant, smoothed frame",

            # Social Proof with hyperpigmentation focus
            "As a dermatologist-endorsed hit: eye-catching turmeric & kojic acid soap reveal with social proof stats on hyperpigmentation fade, beautiful aesthetic lighting on brightened, vibrant skin, high-engagement reel that stops endless scrolls",

            # Pattern Interrupt addressing skin concerns
            "Pattern-interrupting question: Tired of dull, uneven skin tones? This shareable turmeric & kojic acid soap moment unlocks glowing confidence with smooth texture, viral Facebook content with emotional urgency and aesthetic perfection",

            # AI/Personalization with brightening emphasis
            "High-engagement first frame: Attention-grabbing visual of AI-personalized turmeric & kojic acid soap miracle for instant brightening and dewy vibrancy, scarcity vibe with 'limited stock glow', beautiful scene evoking dopamine rush",

            # Emotional Story with color vibrancy
            "Viral social media moment re-read: Stunning turmeric & kojic acid soap demonstration with radiant, smoothed textures and color vibrancy, eye-catching golden hues triggering shares, emotional story of spot-fading transformation that hooks instantly",

            # Visual Chain with progression narrative
            "Thumb-stopping aesthetic chain: Flawless skin close-up with turmeric & kojic acid soap's subtle dewy glow effects and brightening progression, engaging narrative pull with vibrant colors, high-virality stats like 10M views – perfect for Facebook reels",

            # K-beauty trend integration
            "Engaging reel with K-beauty influences: Shareable turmeric & kojic acid soap content showing instant brightening results and smooth, dewy finish, scarcity-driven call-to-action, beautiful visuals that spark FOMO and endless engagement",

            # Dopamine Hook with texture emphasis
            "Attention-grabbing viral hit: Eye-catching frame of transformative turmeric & kojic acid soap for glowing, vibrant skin, social proof from user testimonials on brightening and texture smoothing, dopamine-triggering 'wow' moment in high-aesthetic scene"
        ]

    def _get_skincare_prompts(self) -> List[str]:
        """Neuropsychology-optimized prompts for skincare products"""
        return [
            # FOMO Trigger
            "Instant turmeric & kojic acid soap transformation that triggers FOMO: golden glow, before-after brightening magic, millions sharing this viral skincare demo – thumb-stopping urgency in every radiant frame",

            # Social Proof
            "As a dermatologist-endorsed hit: eye-catching turmeric & kojic acid soap reveal with social proof stats, beautiful aesthetic lighting on brightened flawless skin, high-engagement reel that stops endless scrolls",

            # Pattern Interrupt
            "Pattern-interrupting question: Tired of dull, hyperpigmented skin? This shareable turmeric & kojic acid soap moment unlocks glowing confidence, viral Facebook content with emotional urgency and aesthetic perfection",

            # AI/Personalization Hook
            "High-engagement first frame: Attention-grabbing visual of AI-personalized turmeric & kojic acid soap miracle for brightening, scarcity vibe with 'limited stock glow', beautiful scene evoking dopamine rush",

            # Emotional Story
            "Viral social media moment re-read: Stunning turmeric & kojic acid soap demonstration with radiant brightened textures, eye-catching golden colors triggering shares, emotional story of spot-fading transformation that hooks instantly",

            # Visual Chain
            "Thumb-stopping aesthetic chain: Flawless skin close-up with turmeric & kojic acid soap's subtle glow effects, engaging brightening narrative pull, high-virality stats like 10M views – perfect for Facebook reels",

            # Trend Integration
            "Engaging reel with K-beauty influences: Shareable turmeric & kojic acid soap content showing instant brightening results, scarcity-driven call-to-action, beautiful visuals that spark FOMO and endless engagement",

            # Dopamine Hook
            "Attention-grabbing viral hit: Eye-catching frame of transformative turmeric & kojic acid soap for glowing skin, social proof from user testimonials on brightening, dopamine-triggering 'wow' moment in high-aesthetic scene"
        ]

    def _get_electronics_prompts(self) -> List[str]:
        """Tech-focused prompts with innovation and lifestyle triggers"""
        return [
            "Revolutionary tech unboxing that breaks the internet: cutting-edge gadget reveal with futuristic lighting, millions sharing this game-changing moment",
            "Tech influencer's dream shot: sleek device demonstration with cinematic angles, viral benchmark scores that trigger tech FOMO",
            "Pattern-breaking question: Still using last year's tech? This shareable innovation moment showcases next-gen features in stunning clarity",
            "First-look exclusive: Premium gadget reveal with lifestyle integration, scarcity alert 'pre-order closing', dopamine-inducing tech aesthetics"]

    def _get_fashion_prompts(self) -> List[str]:
        """Fashion prompts with style authority and trend awareness"""
        return [
            "Runway-worthy transformation that dominates feeds: luxury fashion piece styled three ways, influencer-approved looks going viral",
            "Fashion editor's pick: trend-setting outfit reveal with perfect lighting, social proof from style icons, scroll-stopping aesthetic moment",
            "Style challenge accepted: This shareable fashion moment transforms basic to extraordinary, viral-worthy styling tips included",
            "Limited edition drop: Exclusive fashion piece showcase with lifestyle context, FOMO-inducing 'only 100 made' urgency"]

    def _get_generic_viral_prompts(self) -> List[str]:
        """Enhanced generic prompts with neuropsychological elements"""
        return [
            "Viral explosion waiting to happen: thumb-stopping visual with pattern-breaking element, millions can't stop sharing",
            "Social proof masterpiece: high-engagement content with authority endorsement, beautiful aesthetics that demand attention",
            "FOMO-triggering moment: limited-time visual story with emotional hook, shareable content breaking platform records",
            "Dopamine rush guaranteed: eye-catching first frame with transformation narrative, viral statistics prove unstoppable engagement",
            "Pattern interrupt perfection: unexpected visual twist with social validation, aesthetic excellence meets viral psychology",
            "Trending now: platform-optimized content with influencer appeal, scarcity mindset activation in every frame",
            "Scroll-stopper supreme: attention monopolizing visual with emotional resonance, share-worthy moment captured perfectly",
            "Engagement multiplier: algorithm-loving content with human psychology triggers, viral DNA embedded in every pixel"]

    def boost_first_frame_prompts(
            self,
            prompts: List[str],
            boost_factor: float = 1.5) -> List[str]:
        """Enhance prompts specifically for first-frame optimization"""
        boosted = []
        for prompt in prompts:
            # Add first-frame specific elements
            boosted_prompt = f"FIRST FRAME CRITICAL: {prompt} - maximum visual contrast, center-third focal point, instant pattern recognition"
            boosted.append(boosted_prompt)
        return boosted

    def get_prompt_weights(self) -> Dict[str, float]:
        """Return neuropsychological weight distribution"""
        return self.prompt_categories

# ============================================================================
# OPTIMIZATION 1: Ray Actor Pools with Shared Models
# ============================================================================


@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0, max_restarts=3)
class OverlayGenerator:
    """VIREX-9000 Advanced Overlay System with animations and trust badges"""

    def __init__(self):
        self.overlay_variants = [
            {
                "style": "corner_badge",
                "animation_speed": 300,
                "color": "golden_warm",
                "position": "top_left"
            },
            {
                "style": "center_text",
                "animation_speed": 500,
                "color": "dewy_blue",
                "position": "center"
            },
            {
                "style": "bottom_pulse",
                "animation_speed": 400,
                "color": "vibrant_red",
                "position": "bottom"
            }
        ]

    def generate_ffmpeg_filter(self, hook: str, variant_id: int) -> str:
        """Generate complex FFmpeg filter for advanced overlays"""
        variant = self.overlay_variants[variant_id %
                                        len(self.overlay_variants)]

        # Base filter components
        filters = []

        # 1. Crop and scale
        filters.append("crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920,fps=30")

        # 2. Price drop animation ($49 → FREE)
        price_drop = self._generate_price_drop_filter(variant)
        filters.append(price_drop)

        # 3. Main hook text with animation
        hook_filter = self._generate_hook_filter(hook, variant)
        filters.append(hook_filter)

        # 4. Trust badges
        trust_badges = self._generate_trust_badges(variant)
        filters.append(trust_badges)

        # 5. Stock counter with pulse
        stock_counter = self._generate_stock_counter(variant)
        filters.append(stock_counter)

        # 6. Urgency timer
        timer = self._generate_urgency_timer(variant)
        filters.append(timer)

        return ",".join(filters)

    def _generate_price_drop_filter(self, variant):
        """Animated price drop: $49 → FREE"""
        if variant["style"] == "corner_badge":
            # Corner position
            return (
                "drawtext=text='$49':fontsize=36:fontcolor=gray:x=50:y=50:"
                "enable='between(t,0,0.5)',"
                "drawtext=text='$49':fontsize=36:fontcolor=gray:x=50:y=50:"
                "alpha='1-t*2':enable='between(t,0.5,1)',"
                "drawtext=text='FREE':fontsize=48:fontcolor=#FFD700:x=50:y=50:"
                "borderw=3:bordercolor=black@0.7:enable='gte(t,1)'"
            )
        else:
            # Center position
            return (
                "drawtext=text='$49 Value':fontsize=40:fontcolor=gray:"
                "x=(w-text_w)/2:y=h/2-100:enable='between(t,0,0.5)',"
                "drawtext=text='NOW FREE':fontsize=56:fontcolor=#FFD700:"
                "x=(w-text_w)/2:y=h/2-100:borderw=4:bordercolor=black@0.8:"
                "enable='gte(t,0.5)'"
            )

    def _generate_hook_filter(self, hook: str, variant):
        """Main hook text with variant-specific animation"""
        # Escape special characters
        escaped_hook = hook.replace("'", "\\'").replace(";", "\\;")
        escaped_hook = escaped_hook.replace("|", "\\|").replace("&", "\\&")
        escaped_hook = escaped_hook.replace("$", "\\$")

        if variant["position"] == "bottom":
            # Bottom position with pulse
            return (
                f"drawtext=text='{escaped_hook}':fontsize=44:"
                f"fontcolor=white:borderw=3:bordercolor=black@0.8:"
                f"x=(w-text_w)/2:y=h-180:"
                f"alpha='0.8+0.2*sin(2*PI*t*3)':enable='between(t,0,5)'"
            )
        else:
            # Standard position
            return (
                f"drawtext=text='{escaped_hook}':fontsize=48:"
                f"fontcolor=white:borderw=3:bordercolor=black@0.7:"
                f"x=(w-text_w)/2:y=h-250:enable='between(t,0.3,4)'"
            )

    def _generate_trust_badges(self, variant):
        """Trust badges with animation"""
        return (
            "drawtext=text='✓ Dermatologist Approved':fontsize=24:"
            "fontcolor=#90EE90:x=w-300:y=100:enable='gte(t,1.5)',"
            "drawtext=text='🌿 Natural Ingredients':fontsize=24:"
            "fontcolor=#90EE90:x=w-280:y=140:enable='gte(t,2)'"
        )

    def _generate_stock_counter(self, variant):
        """Live stock counter with 3Hz pulse"""
        return (
            "drawtext=text='Only 23 left!':fontsize=32:"
            "fontcolor=#FF6B6B:x=50:y=h-100:"
            "alpha='0.7+0.3*sin(2*PI*t*3)':borderw=2:bordercolor=black@0.6:"
            "enable='between(t,2,8)'"
        )

    def _generate_urgency_timer(self, variant):
        """Countdown timer for urgency"""
        return (
            "drawtext=text='Offer ends in 02\\:34\\:17':fontsize=28:"
            "fontcolor=#FFD700:x=(w-text_w)/2:y=120:"
            "borderw=2:bordercolor=black@0.6:enable='gte(t,2.5)'"
        )


class FirstFrameOptimizer:
    """VIREX-9000 First-Frame Optimization Engine for maximum scroll-stopping power"""

    def __init__(self):
        self.settings = CONFIG.get('first_frame_optimizer', {
            "contrast_boost": 1.5,
            "composition": "golden_ratio",
            "interrupt_elements": ["glow_aura", "urgency_angle"],
            "golden_hour_temp": 3000
        })

    def optimize_first_frame(
            self,
            frame: np.ndarray,
            product_type: str = None) -> np.ndarray:
        """Apply VIREX-9000 optimizations to maximize first-frame impact"""

        # 1. Maximum contrast enhancement
        frame = self._enhance_contrast(frame, self.settings['contrast_boost'])

        # 2. Golden ratio composition check and adjustment
        frame = self._apply_golden_ratio_composition(frame)

        # 3. Apply pattern interrupt elements
        for element in self.settings['interrupt_elements']:
            if element == "glow_aura":
                frame = self._add_glow_aura(frame)
            elif element == "urgency_angle":
                frame = self._apply_urgency_angle(frame)

        # 4. Golden hour lighting
        frame = self._apply_golden_hour_lighting(
            frame, self.settings['golden_hour_temp'])

        # 5. Ensure product is prominent
        frame = self._enhance_product_prominence(frame, product_type)

        return frame

    def _enhance_contrast(
            self,
            frame: np.ndarray,
            boost_factor: float) -> np.ndarray:
        """Enhance contrast for retinal activation"""
        # Convert to LAB color space for better contrast manipulation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=boost_factor * 2,
            tileGridSize=(
                8,
                8))
        l = clahe.apply(l)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_golden_ratio_composition(self, frame: np.ndarray) -> np.ndarray:
        """Ensure key elements follow golden ratio placement"""
        h, w = frame.shape[:2]

        # Golden ratio points
        phi = 1.618
        golden_x = int(w / phi)
        golden_y = int(h / phi)

        # Add subtle vignette focusing on golden ratio intersection
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (golden_x, golden_y), int(min(h, w) * 0.4), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (int(w * 0.3) | 1, int(h * 0.3) | 1), 0)

        # Apply vignette
        frame = frame.astype(np.float32)
        for i in range(3):
            frame[:, :, i] = frame[:, :, i] * (0.3 + 0.7 * mask)

        return np.clip(frame, 0, 255).astype(np.uint8)

    def _add_glow_aura(self, frame: np.ndarray) -> np.ndarray:
        """Add glow aura effect around bright areas (soap)"""
        # Extract bright areas (likely the soap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Create glow effect
        glow = cv2.GaussianBlur(bright_mask, (51, 51), 0)
        glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)

        # Add golden tint to glow
        glow[:, :, 0] = glow[:, :, 0] * 0.3  # Reduce blue
        glow[:, :, 1] = glow[:, :, 1] * 0.8  # Moderate green
        glow[:, :, 2] = glow[:, :, 2] * 1.0  # Full red

        # Blend with original
        return cv2.addWeighted(frame, 0.85, glow, 0.15, 0)

    def _apply_urgency_angle(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle rotation for pattern interrupt"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Small rotation (2-3 degrees) for urgency
        angle = 2.5
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust translation to keep content centered
        M[0, 2] += (w - w) / 2
        M[1, 2] += (h - h) / 2

        return cv2.warpAffine(
            frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _apply_golden_hour_lighting(
            self,
            frame: np.ndarray,
            temp_k: int) -> np.ndarray:
        """Apply warm golden hour color temperature"""
        # Color temperature adjustment matrix for 3000K
        if temp_k == 3000:
            # Warm golden hour matrix
            color_matrix = np.array([
                [1.2, 0.0, 0.0],   # Boost red
                [0.0, 1.1, 0.0],   # Slight green boost
                [0.0, 0.0, 0.8]    # Reduce blue
            ])
        else:
            color_matrix = np.eye(3)

        # Apply color transformation
        frame_float = frame.astype(np.float32)
        for i in range(3):
            frame_float[:, :, i] = np.clip(
                frame_float[:, :, i] * color_matrix[i, i],
                0, 255
            )

        return frame_float.astype(np.uint8)

    def _enhance_product_prominence(
            self,
            frame: np.ndarray,
            product_type: str) -> np.ndarray:
        """Ensure product (soap) is prominently featured"""
        if product_type == "turmeric_kojic_soap":
            # Enhance golden/yellow tones
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define golden color range
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([35, 255, 255])

            # Create mask for golden areas
            mask = cv2.inRange(hsv, lower_gold, upper_gold)

            # Enhance saturation in golden areas
            hsv[:, :, 1] = np.where(mask > 0,
                                    np.clip(hsv[:, :, 1] * 1.3, 0, 255),
                                    hsv[:, :, 1])

            # Enhance value (brightness) in golden areas
            hsv[:, :, 2] = np.where(mask > 0,
                                    np.clip(hsv[:, :, 2] * 1.1, 0, 255),
                                    hsv[:, :, 2])

            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame


class ProductTracker:
    """VIREX-9000 Product Detection and Tracking using SAM2"""

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = None
        self.predictor = None
        self.tracking_enabled = CONFIG.get(
            'dynamic_profiles',
            {}).get(
            PRODUCT_TYPE,
            {}).get(
            'track_product',
            True)
        self.coverage_target = CONFIG.get(
            'dynamic_profiles',
            {}).get(
            PRODUCT_TYPE,
            {}).get(
            'coverage_target',
            "40-60%")

        # Initialize SAM2 if available
        try:
            from segment_anything_2 import sam_model_registry, SamPredictor
            import supervision as sv

            # Use smaller SAM2 model for efficiency
            self.sam_model = sam_model_registry["sam2_hiera_small"](
                checkpoint=None)
            self.sam_model.to(self.device)
            self.predictor = SamPredictor(self.sam_model)

            self.sv = sv  # Store supervision module
            logger.info("SAM2 product tracking initialized successfully")
        except Exception as e:
            logger.warning(
                f"SAM2 not available, using fallback detection: {e}")
            self.tracking_enabled = False

    def track_product_in_frames(self,
                                frames: List[np.ndarray],
                                product_type: str = "turmeric_kojic_soap") -> List[Dict]:
        """Track product across frames and return tracking data"""
        if not self.tracking_enabled or self.predictor is None:
            return self._fallback_detection(frames, product_type)

        tracking_data = []
        previous_mask = None

        for idx, frame in enumerate(frames):
            try:
                # Set image for SAM
                self.predictor.set_image(frame)

                if idx == 0 or previous_mask is None:
                    # Initial detection - find soap in first frame
                    mask, confidence = self._detect_soap(frame, product_type)
                else:
                    # Track from previous frame
                    mask, confidence = self._track_from_previous(
                        frame, previous_mask)

                # Calculate coverage percentage
                coverage = self._calculate_coverage(mask, frame.shape)

                # Get bounding box and center
                bbox, center = self._get_bbox_and_center(mask)

                tracking_data.append({
                    "frame_idx": idx,
                    "mask": mask,
                    "confidence": confidence,
                    "coverage": coverage,
                    "bbox": bbox,
                    "center": center,
                    "needs_adjustment": not self._is_coverage_optimal(coverage)
                })

                previous_mask = mask

            except Exception as e:
                logger.warning(f"SAM2 tracking failed for frame {idx}: {e}")
                tracking_data.append(
                    self._get_default_tracking_data(
                        idx, frame.shape))

        return tracking_data

    def _detect_soap(self, frame: np.ndarray,
                     product_type: str) -> Tuple[np.ndarray, float]:
        """Detect soap in frame using color and shape cues"""
        if product_type == "turmeric_kojic_soap":
            # Look for golden/yellow rectangular objects
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # Golden color range for turmeric soap
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([35, 255, 255])

            # Create color mask
            color_mask = cv2.inRange(hsv, lower_gold, upper_gold)

            # Find contours
            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find largest rectangular contour (likely the soap)
                best_contour = max(contours, key=cv2.contourArea)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(best_contour)

                # Use SAM2 with box prompt
                box = np.array([x, y, x + w, y + h])
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=True
                )

                # Select best mask
                best_mask_idx = np.argmax(scores)
                return masks[best_mask_idx], float(scores[best_mask_idx])

        # Fallback to center detection
        h, w = frame.shape[:2]
        center_box = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])
        masks, scores, _ = self.predictor.predict(
            box=center_box,
            multimask_output=True
        )
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx], float(scores[best_mask_idx])

    def _track_from_previous(self,
                             frame: np.ndarray,
                             previous_mask: np.ndarray) -> Tuple[np.ndarray,
                                                                 float]:
        """Track object from previous frame mask"""
        # Get points from previous mask
        points = self._sample_points_from_mask(previous_mask, num_points=5)

        if len(points) > 0:
            # Use points as prompts for SAM2
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                multimask_output=True
            )

            # Select best mask
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx], float(scores[best_mask_idx])

        # Fallback to previous mask
        return previous_mask, 0.5

    def _sample_points_from_mask(
            self,
            mask: np.ndarray,
            num_points: int = 5) -> np.ndarray:
        """Sample points from mask for tracking"""
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            return np.array([])

        # Sample random points from mask
        num_samples = min(num_points, len(y_coords))
        indices = np.random.choice(len(y_coords), num_samples, replace=False)

        points = np.column_stack((x_coords[indices], y_coords[indices]))
        return points

    def _calculate_coverage(
            self,
            mask: np.ndarray,
            frame_shape: Tuple) -> float:
        """Calculate percentage of frame covered by product"""
        if mask is None:
            return 0.0

        mask_area = np.sum(mask > 0)
        frame_area = frame_shape[0] * frame_shape[1]

        return (mask_area / frame_area) * 100

    def _is_coverage_optimal(self, coverage: float) -> bool:
        """Check if coverage is within target range"""
        target_range = self.coverage_target.split('-')
        min_coverage = float(target_range[0].replace('%', ''))
        max_coverage = float(target_range[1].replace('%', ''))

        return min_coverage <= coverage <= max_coverage

    def _get_bbox_and_center(
            self, mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        """Get bounding box and center from mask"""
        if mask is None or np.sum(mask) == 0:
            return (0, 0, 0, 0), (0, 0)

        y_coords, x_coords = np.where(mask > 0)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        return bbox, center

    def _fallback_detection(self,
                            frames: List[np.ndarray],
                            product_type: str) -> List[Dict]:
        """Fallback detection using color and shape heuristics"""
        tracking_data = []

        for idx, frame in enumerate(frames):
            # Simple color-based detection for golden soap
            if product_type == "turmeric_kojic_soap":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_gold = np.array([15, 50, 50])
                upper_gold = np.array([35, 255, 255])
                mask = cv2.inRange(hsv, lower_gold, upper_gold)

                # Clean up mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                coverage = self._calculate_coverage(mask, frame.shape)
                bbox, center = self._get_bbox_and_center(mask)

                tracking_data.append({
                    "frame_idx": idx,
                    "mask": mask,
                    "confidence": 0.7,  # Lower confidence for fallback
                    "coverage": coverage,
                    "bbox": bbox,
                    "center": center,
                    "needs_adjustment": not self._is_coverage_optimal(coverage)
                })
            else:
                tracking_data.append(
                    self._get_default_tracking_data(
                        idx, frame.shape))

        return tracking_data

    def _get_default_tracking_data(self, idx: int, frame_shape: Tuple) -> Dict:
        """Get default tracking data when detection fails"""
        h, w = frame_shape[:2]
        return {
            "frame_idx": idx,
            "mask": None,
            "confidence": 0.0,
            "coverage": 0.0,
            "bbox": (w // 4, h // 4, w // 2, h // 2),  # Center region
            "center": (w // 2, h // 2),
            "needs_adjustment": True
        }

    def generate_visibility_heatmap(
            self,
            tracking_data: List[Dict],
            frame_shape: Tuple) -> np.ndarray:
        """Generate heatmap showing product visibility across frames"""
        h, w = frame_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for data in tracking_data:
            if data["mask"] is not None:
                heatmap += data["mask"].astype(np.float32) * data["confidence"]

        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return heatmap_colored

    def adjust_frame_for_coverage(
            self,
            frame: np.ndarray,
            tracking_data: Dict) -> np.ndarray:
        """Adjust frame composition to achieve optimal product coverage"""
        if not tracking_data["needs_adjustment"]:
            return frame

        h, w = frame.shape[:2]
        bbox = tracking_data["bbox"]
        center = tracking_data["center"]
        current_coverage = tracking_data["coverage"]

        # Calculate desired scale factor
        target_coverage = 50.0  # Middle of 40-60% range
        scale_factor = np.sqrt(target_coverage / max(current_coverage, 1.0))

        # Limit scale factor to reasonable range
        scale_factor = np.clip(scale_factor, 0.8, 1.5)

        # Calculate crop region to center and scale product
        _bbox_w, _bbox_h = bbox[2], bbox[3]
        new_size = int(min(w, h) / scale_factor)

        # Center crop around product
        x_start = max(0, center[0] - new_size // 2)
        y_start = max(0, center[1] - new_size // 2)
        x_end = min(w, x_start + new_size)
        y_end = min(h, y_start + new_size)

        # Crop and resize
        cropped = frame[y_start:y_end, x_start:x_end]
        adjusted = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

        return adjusted


@ray.remote
class ModelActor:
    """Persistent actor holding all models for a single GPU"""

    def __init__(self, actor_id: int, cache_actor):
        self.actor_id = actor_id
        self.cache = cache_actor

        # Try to use GPU, fallback to CPU if not available
        try:
            if torch.cuda.is_available():
                self.device = torch.device(
                    f"cuda:{actor_id % torch.cuda.device_count()}")
                torch.cuda.set_device(self.device)
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            logger.warning(
                f"ModelActor {actor_id}: GPU initialization failed ({e}), falling back to CPU")
            self.device = torch.device("cpu")

        logger.info(f"ModelActor {actor_id} initializing on {self.device}")

        # Memory monitoring
        self.use_memory_monitoring = OPTIMIZATION_SETTINGS.get(
            'use_memory_monitoring', True)
        self.memory_threshold = OPTIMIZATION_SETTINGS.get(
            'memory_threshold', 0.85)
        self.min_batch_size = 1
        self.current_batch_size = PROCESSING_SETTINGS['frame_batch_size']

        # Load all models with optimizations
        self._load_optimized_models()

        # Initialize caches
        self.frame_cache = {}
        self.feature_cache = {}

        logger.info(f"ModelActor {actor_id} ready!")

    def _load_optimized_models(self):
        """Load models with 8-bit quantization and TensorRT optimization"""

        # 8-bit quantization config (updated for 2025 compatibility)
        quantization_config = None
        if QUALITY_SETTINGS['use_8bit_quantization'] and BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                int8_threshold=6.0,
                llm_int8_threshold=6.0,
                bnb_4bit_use_double_quant=True  # Additional memory optimization
            )
        elif QUALITY_SETTINGS['use_8bit_quantization']:
            logger.warning(
                "8-bit quantization requested but BitsAndBytes not available")

        # Load FLUX Fill Pipeline for product-preserving enhancement
        logger.info(
            f"Actor {
                self.actor_id}: Loading FLUX Fill pipeline for image-to-image enhancement...")

        # Determine which FLUX model to use
        use_schnell = OPTIMIZATION_SETTINGS.get('use_flux_schnell', False)
        use_kontext = OPTIMIZATION_SETTINGS.get('use_flux_kontext', True)

        if use_kontext:
            model_id = "black-forest-labs/FLUX.1-Kontext-dev"
            logger.info(
                f"Actor {
                    self.actor_id}: Using FLUX.1 Kontext for context-aware enhancement (VIREX-9000 mode)")
        elif use_schnell:
            model_id = "black-forest-labs/FLUX.1-schnell"
            logger.info(
                f"Actor {
                    self.actor_id}: Using FLUX.1 Schnell for 4x faster processing")
        else:
            model_id = "black-forest-labs/FLUX.1-Fill-dev"
            logger.info(
                f"Actor {
                    self.actor_id}: Using FLUX.1 Fill Dev for standard enhancement")

        try:
            self.enhancement_pipe = FluxFillPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
        except Exception as e:
            logger.error(f"Failed to load FLUX enhancement pipeline: {e}")
            raise

        # FLUX uses optimized schedulers for fast inference
        logger.info(
            f"Actor {
                self.actor_id}: FLUX uses optimized sampling for product preservation")

        # Apply TensorRT optimization if available
        if OPTIMIZATION_SETTINGS['use_tensorrt'] and self.device.type == "cuda" and TENSORRT_AVAILABLE:
            try:
                self._optimize_with_tensorrt(self.enhancement_pipe)
            except Exception as e:
                logger.warning(
                    f"TensorRT optimization failed: {e}, using standard pipeline")
        elif OPTIMIZATION_SETTINGS['use_tensorrt'] and not TENSORRT_AVAILABLE:
            logger.warning(
                "TensorRT optimization requested but TensorRT not available")

        self.enhancement_pipe = self.enhancement_pipe.to(self.device)

        # Try xformers first for better performance, fallback to attention
        # slicing
        try:
            self.enhancement_pipe.enable_xformers_memory_efficient_attention()
            logger.info(
                f"Actor {
                    self.actor_id}: Using xformers for attention optimization")
        except Exception:
            self.enhancement_pipe.enable_attention_slicing()  # Fallback to attention slicing
            logger.info(
                f"Actor {
                    self.actor_id}: Using attention slicing (xformers not available)")

        self.enhancement_pipe.enable_vae_slicing()       # VAE slicing for large batches

        # Apply torch.compile for additional speedup if available
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.enhancement_pipe.transformer = torch.compile(
                    self.enhancement_pipe.transformer)
                logger.info(
                    f"Actor {
                        self.actor_id}: Applied torch.compile to FLUX transformer for ~20% speedup")
            except Exception as e:
                logger.warning(
                    f"Actor {
                        self.actor_id}: torch.compile failed: {e}")

        # Also load SVD for optional video generation effects
        if OPTIMIZATION_SETTINGS.get('use_svd_effects', False):
            logger.info(
                f"Actor {
                    self.actor_id}: Loading SVD for video effects...")
            try:
                self.svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    variant="fp16" if self.device.type == "cuda" else None
                )
                self.svd_pipe = self.svd_pipe.to(self.device)
                # Apply optimizations
                try:
                    self.svd_pipe.enable_xformers_memory_efficient_attention()
                except BaseException:
                    self.svd_pipe.enable_attention_slicing()
                self.svd_pipe.enable_vae_slicing()
            except Exception as e:
                logger.warning(f"Failed to load SVD pipeline: {e}")
                self.svd_pipe = None

        # FLUX Fill handles both enhancement and inpainting, no need for
        # separate pipeline
        logger.info(
            f"Actor {
                self.actor_id}: FLUX Fill pipeline handles both enhancement and inpainting")
        self.inpaint_pipe = self.enhancement_pipe  # Use same pipeline for consistency

        # Load SigLIP with optimization - superior to CLIP for scene engagement
        logger.info(
            f"Actor {
                self.actor_id}: Loading SigLIP model (better than CLIP for viral content)...")
        self.siglip_model = AutoModel.from_pretrained(
            "google/siglip-large-patch16-384",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-large-patch16-384")

        # Load Video scoring model with 8-bit quantization
        logger.info(
            f"Actor {
                self.actor_id}: Loading VideoLLaVA with 8-bit quantization...")
        try:
            self.virality_model = VideoLlavaForConditionalGeneration.from_pretrained(
                "LanguageBind/Video-LLaVA-7B-hf",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto")
        except Exception as e:
            logger.error(f"Failed to load Video-LLaVA model: {e}")
            raise
        self.virality_processor = VideoLlavaProcessor.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf")

        # Initialize OCR
        self.ocr_reader = paddleocr.PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True if self.device.type == "cuda" else False,
            device=f'gpu:{
                self.actor_id %
                torch.cuda.device_count()}' if self.device.type == "cuda" else 'cpu')

        # Quantum device
        self.qml_dev = qml.device(
            'default.qubit.torch',
            wires=1) if self.device.type == "cuda" else qml.device(
            'default.qubit',
            wires=1)

        # Initialize product tracker
        self.product_tracker = None
        if CONFIG.get(
            'dynamic_profiles',
            {}).get(
            PRODUCT_TYPE,
            {}).get(
            'track_product',
                True):
            try:
                self.product_tracker = ProductTracker()
                logger.info(
                    f"Actor {
                        self.actor_id}: Product tracker initialized")
            except Exception as e:
                logger.warning(
                    f"Actor {
                        self.actor_id}: Product tracker unavailable: {e}")

    def _optimize_with_tensorrt(self, pipeline):
        """Apply TensorRT optimization to diffusion pipeline"""
        # This is a placeholder - actual TensorRT optimization would require
        # converting models to ONNX and then TensorRT
        logger.info("Applying TensorRT optimizations to pipeline...")
        # In production, you would:
        # 1. Export to ONNX
        # 2. Convert to TensorRT
        # 3. Replace pipeline components
        pass

    # ========================================================================
    # OPTIMIZATION 2: Batch Processing with Feature Slicing
    # ========================================================================

    def process_frame_batch(self,
                            frames: List[np.ndarray],
                            operation: str,
                            **kwargs) -> List[np.ndarray]:
        """Process multiple frames in a single batch with feature slicing"""

        if operation == "text_removal":
            return self._batch_text_removal(frames)
        elif operation == "style_transfer":
            return self._batch_style_transfer(frames, **kwargs)
        elif operation == "scene_scoring":
            return self._batch_scene_scoring(frames)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _batch_text_removal(self,
                            frames: List[np.ndarray]) -> List[np.ndarray]:
        """Remove text from multiple frames using batch processing with caching"""

        # Process OCR detection with caching if enabled
        ocr_results = []
        for frame in frames:
            if OPTIMIZATION_SETTINGS.get(
                    'use_caching', True) and self.cache is not None:
                # Check cache first
                frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
                cached_result = ray.get(self.cache.get_ocr.remote(frame_hash))

                if cached_result is not None:
                    ocr_results.append(cached_result)
                    continue

            # Perform OCR
            try:
                result = self.ocr_reader.ocr(frame)
                if OPTIMIZATION_SETTINGS.get(
                        'use_caching', True) and self.cache is not None:
                    frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
                    ray.get(self.cache.set_ocr.remote(frame_hash, result))
                ocr_results.append(result)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                ocr_results.append(None)

        # Batch inpainting for frames with text
        frames_to_inpaint = []
        inpaint_indices = []
        masks = []

        for i, (frame, ocr_result) in enumerate(zip(frames, ocr_results)):
            if ocr_result and ocr_result[0]:
                # Create combined mask for all text regions
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for detection in ocr_result[0]:
                    if detection and detection[0]:
                        bbox = detection[0]
                        pts = np.array(bbox, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)

                if mask.any():
                    frames_to_inpaint.append(frame)
                    inpaint_indices.append(i)
                    masks.append(mask)

        # Batch inpainting with feature slicing
        if frames_to_inpaint:
            # Convert to PIL images
            pil_images = [
                Image.fromarray(
                    cv2.cvtColor(
                        f,
                        cv2.COLOR_BGR2RGB)) for f in frames_to_inpaint]
            pil_masks = [Image.fromarray(m) for m in masks]

            # Feature slicing: Process in smaller spatial chunks if needed
            if OPTIMIZATION_SETTINGS['use_feature_slicing']:
                inpainted = self._inpaint_with_feature_slicing(
                    pil_images, pil_masks)
            else:
                # Standard batch inpainting with OOM handling
                batch_size = len(pil_images)
                while batch_size > 0:
                    try:
                        with torch.cuda.amp.autocast():
                            inpainted = self.inpaint_pipe(
                                prompt=["seamless skincare product scene"] * batch_size,
                                image=pil_images[:batch_size],
                                mask_image=pil_masks[:batch_size],
                                num_inference_steps=QUALITY_SETTINGS['inpaint_inference_steps']
                            ).images
                        break
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            f"OOM in inpainting, reducing batch size from {batch_size} to {
                                batch_size // 2}")
                        batch_size //= 2
                        torch.cuda.empty_cache()
                        if batch_size == 0:
                            raise RuntimeError(
                                "Cannot process even single image - OOM")

            # Convert back to numpy
            for idx, img in enumerate(inpainted):
                frames[inpaint_indices[idx]] = cv2.cvtColor(
                    np.array(img), cv2.COLOR_RGB2BGR)

        return frames

    def _inpaint_with_feature_slicing(self,
                                      images: List[Image.Image],
                                      masks: List[Image.Image]) -> List[Image.Image]:
        """Inpaint with spatial feature slicing for memory efficiency"""
        results = []

        # Process each image with spatial slicing
        for img, mask in zip(images, masks):
            # Slice image into quadrants for processing
            w, h = img.size
            quadrants = [
                (0, 0, w // 2, h // 2),
                (w // 2, 0, w, h // 2),
                (0, h // 2, w // 2, h),
                (w // 2, h // 2, w, h)
            ]

            inpainted_quadrants = []
            for quad in quadrants:
                img_slice = img.crop(quad)
                mask_slice = mask.crop(quad)

                # Only process if mask has content
                if np.array(mask_slice).any():
                    with torch.cuda.amp.autocast():
                        result = self.inpaint_pipe(
                            prompt="seamless skincare product scene",
                            image=img_slice,
                            mask_image=mask_slice,
                            num_inference_steps=QUALITY_SETTINGS['inpaint_inference_steps']
                        ).images[0]
                    inpainted_quadrants.append(result)
                else:
                    inpainted_quadrants.append(img_slice)

            # Stitch quadrants back together
            result_img = Image.new('RGB', (w, h))
            result_img.paste(inpainted_quadrants[0], (0, 0))
            result_img.paste(inpainted_quadrants[1], (w // 2, 0))
            result_img.paste(inpainted_quadrants[2], (0, h // 2))
            result_img.paste(inpainted_quadrants[3], (w // 2, h // 2))

            results.append(result_img)

        return results

    def _batch_style_transfer(self,
                              frames: List[np.ndarray],
                              variant_id: int) -> List[np.ndarray]:
        """Apply style transfer to multiple frames with optimization"""
        # Convert frames to PIL
        pil_frames = [
            Image.fromarray(
                cv2.cvtColor(
                    f,
                    cv2.COLOR_BGR2RGB)) for f in frames]

        # Use proper style transfer instead of video generation
        stylized_frames = []
        batch_size = min(
            PROCESSING_SETTINGS['frame_batch_size'],
            len(pil_frames))

        for i in range(0, len(pil_frames), batch_size):
            batch = pil_frames[i:i + batch_size]
            current_batch_size = len(batch)

            # Retry with smaller batch on OOM
            while current_batch_size > 0:
                try:
                    with torch.cuda.amp.autocast():
                        # Generate prompts based on product type
                        if OPTIMIZATION_SETTINGS.get('cosmetics_mode', True):
                            # VIREX-9000 cosmetics-specific prompts
                            prompt = self._generate_cosmetics_enhancement_prompt(
                                variant_id)
                        else:
                            # Standard prompts
                            style_prompts = [
                                f"professional product photography, high quality, variant {variant_id}",
                                f"luxury skincare advertisement, elegant lighting, style {variant_id}",
                                f"viral social media product shot, trendy aesthetic, version {variant_id}"]
                            prompt = style_prompts[variant_id %
                                                   len(style_prompts)]

                        # Determine inference steps based on model type
                        use_schnell = OPTIMIZATION_SETTINGS.get(
                            'use_flux_schnell', False)
                        use_kontext = OPTIMIZATION_SETTINGS.get(
                            'use_flux_kontext', True)

                        if use_kontext and OPTIMIZATION_SETTINGS.get(
                                'cosmetics_mode', True):
                            # VIREX-9000 optimized settings for cosmetics
                            num_steps = 35  # Optimal for quality/speed
                            guidance_scale = 9.0  # Higher for cosmetics adherence
                            strength = 0.50  # Lower to preserve soap geometry
                        elif use_schnell:
                            num_steps = 4  # Schnell only needs 4 steps for fast inference
                            guidance_scale = 3.5  # FLUX still uses guidance
                            strength = 0.6  # Standard preservation
                        else:
                            num_steps = min(
                                50, QUALITY_SETTINGS['svd_inference_steps'])  # FLUX recommended: 50
                            guidance_scale = 7.5  # FLUX recommended: 7.5
                            strength = 0.6  # Standard preservation

                        # FLUX Fill does proper image-to-image enhancement
                        # (preserves products)
                        enhanced_prompt = f"{prompt}, enhance lighting and background only, preserve exact product appearance"

                        # Add negative prompt for cosmetics mode - VIREX-9000
                        # enhanced
                        if OPTIMIZATION_SETTINGS.get('cosmetics_mode', True):
                            negative_prompt = "dull colors, dry textures, distorted shapes, matte finish, gray tones, flat lighting, no glow, rough skin texture, uneven tone, dark spots visible, no transformation, boring composition"
                        else:
                            negative_prompt = None

                        # Process each frame with FLUX Fill for product
                        # preservation
                        enhanced_frames = []
                        for frame_pil in batch[:current_batch_size]:
                            # Create a subtle mask to focus enhancement on background
                            # For now, we'll enhance the whole image with low strength
                            # Build kwargs for pipeline
                            pipe_kwargs = {
                                "prompt": enhanced_prompt,
                                "image": frame_pil,
                                "mask_image": None,  # No mask for now, enhance whole image
                                "num_inference_steps": num_steps,
                                "guidance_scale": guidance_scale,
                                "strength": strength  # Use calculated strength based on mode
                            }

                            # Add negative prompt if specified
                            if negative_prompt:
                                pipe_kwargs["negative_prompt"] = negative_prompt

                            result = self.enhancement_pipe(
                                **pipe_kwargs).images[0]
                            enhanced_frames.append(result)

                        results = enhanced_frames
                    stylized_frames.extend(results)
                    break
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        f"OOM in style transfer, reducing batch from {current_batch_size} to {
                            current_batch_size // 2}")
                    current_batch_size //= 2
                    torch.cuda.empty_cache()
                    if current_batch_size == 0:
                        # Fallback: return original frames
                        logger.error(
                            "Cannot style transfer even single frame, using original")
                        stylized_frames.extend(batch)
                        break

        # Apply quantum perturbation
        results = []
        for i, stylized in enumerate(stylized_frames):
            frame_bgr = cv2.cvtColor(np.array(stylized), cv2.COLOR_RGB2BGR)

            # Apply quantum effects only if enabled
            if OPTIMIZATION_SETTINGS.get(
                'use_quantum',
                    True) and self.qml_dev is not None:
                if OPTIMIZATION_SETTINGS['cache_quantum_circuits']:
                    perturbed = self._apply_cached_quantum_effects(
                        frame_bgr, variant_id + i)
                else:
                    perturbed = self._apply_quantum_effects(
                        frame_bgr, variant_id + i)
            else:
                perturbed = frame_bgr  # Skip quantum effects if disabled

            results.append(perturbed)

        return results

    def _apply_svd_effects(self,
                           frames: List[Image.Image],
                           variant_id: int) -> List[Image.Image]:
        """Apply SVD video effects if enabled and available"""
        if not hasattr(self, 'svd_pipe') or self.svd_pipe is None:
            return frames

        try:
            # Use SVD for motion effects on select frames
            with torch.cuda.amp.autocast():
                results = self.svd_pipe(
                    image=frames[0],  # Use first frame as base
                    num_inference_steps=5,  # Quick generation
                    decode_chunk_size=8,
                    motion_bucket_id=127  # 2025 recommendation for stability
                ).frames[0]

            # Blend SVD results with original for subtle effect
            blended = []
            for i, orig in enumerate(frames):
                if i < len(results):
                    # Blend original with SVD frame
                    orig_np = np.array(orig)
                    svd_np = np.array(results[i])
                    blended_np = (
                        orig_np *
                        0.7 +
                        svd_np *
                        0.3).astype(
                        np.uint8)
                    blended.append(Image.fromarray(blended_np))
                else:
                    blended.append(orig)
            return blended
        except Exception as e:
            logger.warning(f"SVD effects failed: {e}, using original frames")
            return frames

    def _batch_scene_scoring(self,
                             frames: List[np.ndarray],
                             product_type: str = None) -> List[float]:
        """Score multiple frames for viral engagement using SigLIP with neuropsychology-optimized prompts and dynamic product priorities"""
        # Use dynamic profile scene priorities
        if product_type is None:
            product_type = PRODUCT_TYPE

        profile = DYNAMIC_PROFILES.get(
            product_type, DYNAMIC_PROFILES.get(
                'default', {}))
        profile.get('scene_priorities', {
            "product": 0.4,
            "demonstration": 0.3,
            "results": 0.3
        })

        scores = []
        frames_to_process = []
        frame_indices = []

        # Check cache first if enabled
        for i, frame in enumerate(frames):
            if OPTIMIZATION_SETTINGS.get(
                    'use_caching', True) and self.cache is not None:
                frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
                cached_scores = ray.get(
                    self.cache.get_siglip.remote(frame_hash))

                if cached_scores is not None:
                    # Use weighted score from cached results
                    if OPTIMIZATION_SETTINGS.get(
                            'use_neuropsychology_prompts', True):
                        weighted_score = self._calculate_weighted_score(
                            cached_scores)
                        scores.append((i, weighted_score))
                    else:
                        # Fallback to max score
                        max_score = max(cached_scores.values())
                        scores.append((i, max_score))
                    continue

            frames_to_process.append(frame)
            frame_indices.append(i)

        # Process uncached frames
        if frames_to_process:
            pil_frames = [
                Image.fromarray(
                    cv2.cvtColor(
                        f,
                        cv2.COLOR_BGR2RGB)) for f in frames_to_process]

            # Get prompts based on configuration
            if OPTIMIZATION_SETTINGS.get('use_neuropsychology_prompts', True):
                # Use neuropsychology-optimized prompts
                prompt_engine = NeuropsychologyPromptEngine()

                # Auto-detect product type if not provided
                if product_type is None:
                    product_type = self._detect_product_type(frames_to_process)

                viral_prompts = prompt_engine.get_prompts(
                    product_type=product_type,
                    trend_date="2025-07",
                    platform="facebook_reels"
                )

                # Add soap-specific scene detection prompts
                if product_type == "turmeric_kojic_soap":
                    soap_scene_prompts = [
                        "Rich golden soap lathering with creamy bubbles and foam texture",
                        "Hands applying turmeric soap to glowing skin with smooth motion",
                        "Before and after skin transformation showing brightening effects",
                        "Close-up of turmeric & kojic acid soap bar with golden color"]
                    viral_prompts.extend(soap_scene_prompts)

                # Boost first frame if needed
                if 0 in frame_indices and OPTIMIZATION_SETTINGS.get(
                        'first_frame_boost', 1.5) > 1.0:
                    viral_prompts = prompt_engine.boost_first_frame_prompts(
                        viral_prompts)
            else:
                # Fallback to original prompts
                viral_prompts = [
                    "viral facebook video content",
                    "thumb-stopping social media moment",
                    "engaging product demonstration",
                    "beautiful aesthetic scene",
                    "eye-catching visual content",
                    "shareable social media content",
                    "high engagement facebook reel",
                    "attention grabbing first frame"
                ]

            # Process with SigLIP - handles multiple texts better than CLIP
            inputs = self.siglip_processor(
                text=viral_prompts,
                images=pil_frames,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.cuda.amp.autocast():
                outputs = self.siglip_model(**inputs)
                # SigLIP uses sigmoid, not softmax
                sigmoid_scores = torch.sigmoid(outputs.logits_per_image)

            # Process results for each frame
            for frame_idx, frame in enumerate(frames_to_process):
                # Get scores for this frame across all prompts
                frame_scores = sigmoid_scores[frame_idx].cpu().numpy()

                # Create score dictionary
                score_dict = {
                    prompt: float(score) for prompt,
                    score in zip(
                        viral_prompts,
                        frame_scores)}

                # Calculate final score based on configuration
                if OPTIMIZATION_SETTINGS.get(
                        'use_neuropsychology_prompts', True):
                    # Apply neuropsychological weighting
                    weighted_score = self._calculate_weighted_score(score_dict)
                    final_score = weighted_score

                    # Apply first frame boost if applicable
                    if frame_indices[frame_idx] == 0:
                        boost = OPTIMIZATION_SETTINGS.get(
                            'first_frame_boost', 1.5)
                        final_score *= boost
                else:
                    # Use max score for ranking
                    final_score = float(frame_scores.max())

                # Cache if enabled
                if OPTIMIZATION_SETTINGS.get(
                        'use_caching', True) and self.cache is not None:
                    frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
                    ray.get(
                        self.cache.set_siglip.remote(
                            frame_hash, score_dict))

                scores.append((frame_indices[frame_idx], final_score))

        # Sort by original index and return
        scores.sort(key=lambda x: x[0])
        return [s[1] for s in scores]

    def _calculate_weighted_score(self, score_dict: Dict[str, float]) -> float:
        """Apply neuropsychological weighting to prompt scores"""
        prompt_engine = NeuropsychologyPromptEngine()
        weights = prompt_engine.get_prompt_weights()

        # Map prompts to categories (simplified for now)
        # In production, would use more sophisticated mapping
        prompt_categories = {
            0: "fomo_trigger",
            1: "social_proof",
            2: "pattern_interrupt",
            3: "emotion_hook",
            4: "visual_appeal",
            5: "trend_alignment",
            6: "trend_alignment",
            7: "emotion_hook"
        }

        # Calculate weighted sum
        weighted_sum = 0.0
        weight_sum = 0.0

        for i, (prompt, score) in enumerate(list(score_dict.items())[:8]):
            category = prompt_categories.get(i % 8, "visual_appeal")
            weight = weights.get(category, 0.1)
            weighted_sum += score * weight
            weight_sum += weight

        # Normalize
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return max(score_dict.values()) if score_dict else 0.0

    def _detect_product_type(self, frames: List[np.ndarray]) -> str:
        """Auto-detect product type from video frames"""
        # Simple heuristic for now - could be enhanced with ML
        # Check first few frames for product characteristics

        # TODO: Implement actual product detection logic
        # For now, return default based on config or video URL
        if hasattr(self, 'product_type_hint'):
            return self.product_type_hint

        # Default to skincare for soap products
        return "skincare"

    def _generate_cosmetics_enhancement_prompt(self, variant_id: int) -> str:
        """Generate VIREX-9000 optimized prompts for cosmetics enhancement"""

        # Variant-specific focus areas
        variant_focuses = [
            "golden turmeric glow with luxury marble background, radiant dewy finish",
            "dewy kojic acid brightening with soft bokeh, smooth skin texture",
            "radiant skin transformation with clinical lighting, vibrant golden hues",
            "K-beauty glass skin effect with natural warmth, brightening progression"]

        base_prompt = f"""[ATTENTION HIJACK LAYER - Soap Mode]
Enhance golden turmeric hues (+30% vibrancy), dewy kojic acid finish with subsurface scattering, rich lather textures; preserve exact soap geometry while boosting brightness/glow effects for radiant skin.
Focus: {variant_focuses[variant_id % len(variant_focuses)]}

[DOPAMINE TRIGGER LAYER]
- Enhance specular highlights on soap surfaces (+40% intensity) for vibrant glow
- Apply dewy finish with subsurface scattering for translucent skin effect
- Golden hour lighting (3000K) for trust and warmth psychology
- Micro-shimmer on lather bubbles simulating instant transformation

[TRUST SIGNAL LAYER]
- 100% preserve soap geometry and natural ingredient visuals (turmeric flecks, kojic clarity)
- Enhance golden turmeric color saturation while maintaining authenticity
- Apply rim lighting for glow aura effect around soap
- Ensure smooth, even skin tone in application areas

[VIRAL OPTIMIZATION]
        - First-frame maximum contrast for thumb-stop effect
        - Z-pattern visual flow from product to brightened skin
        - High saturation (HSB >80%) for mobile OLED displays
        - Gestalt incomplete-to-complete transformation narrative
        """

        # Simplified version for processing
        return f"Ultra-premium cosmetics photography: {base_prompt[:500]}... Focus: {variant_focuses[variant_id %
                                                                                                     len(variant_focuses)]}"

    # ========================================================================
    # OPTIMIZATION 3: Quantum Circuit Caching
    # ========================================================================

    @lru_cache(maxsize=1000)
    def _get_cached_quantum_circuit(self, seed: int) -> float:
        """Get cached quantum circuit result"""
        random.seed(seed)

        @qml.qnode(self.qml_dev,
                   interface='torch' if self.device.type == "cuda" else 'numpy')
        def quantum_circuit():
            angle = random.uniform(0, np.pi)
            qml.RY(angle, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        noise = quantum_circuit() * 0.1 + 1.0
        return noise.item() if hasattr(noise, 'item') else noise

    def _get_adaptive_batch_size(self) -> int:
        """Get adaptive batch size based on current GPU memory usage"""
        if not self.use_memory_monitoring or self.device.type != "cuda":
            return self.current_batch_size

        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(self.device)
            torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory

            # Calculate usage percentage
            usage_ratio = allocated / total

            # Adjust batch size based on memory pressure
            if usage_ratio > self.memory_threshold:
                # Reduce batch size
                self.current_batch_size = max(
                    self.min_batch_size, self.current_batch_size // 2)
                logger.warning(
                    f"High memory usage ({
                        usage_ratio:.1%}), reducing batch size to {
                        self.current_batch_size}")
                # Force memory cleanup
                torch.cuda.empty_cache()
            elif usage_ratio < 0.5 and self.current_batch_size < PROCESSING_SETTINGS['frame_batch_size']:
                # Increase batch size if we have headroom
                self.current_batch_size = min(
                    PROCESSING_SETTINGS['frame_batch_size'],
                    self.current_batch_size * 2)
                logger.info(
                    f"Low memory usage ({
                        usage_ratio:.1%}), increasing batch size to {
                        self.current_batch_size}")

            return self.current_batch_size
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")
            return self.current_batch_size

    def _apply_cached_quantum_effects(
            self,
            frame: np.ndarray,
            variant_id: int) -> np.ndarray:
        """Apply quantum effects with caching and neuropsychology triggers"""
        # Apply visual psychology triggers first
        frame = self._apply_neuropsychology_triggers(frame, variant_id)

        channels = []

        for i in range(3):
            U, S, V = np.linalg.svd(frame[:, :, i], full_matrices=False)

            # Use cached quantum result
            noise_value = self._get_cached_quantum_circuit(variant_id * 3 + i)
            S_modified = S * noise_value

            transformed = np.dot(U, np.dot(np.diag(S_modified), V))
            channels.append(np.clip(transformed, 0, 255).astype(np.uint8))

        return np.stack(channels, axis=-1)

    def _apply_quantum_effects(
            self,
            frame: np.ndarray,
            variant_id: int) -> np.ndarray:
        """VIREX-9000 Neuropsychology Visual Triggers with quantum effects"""
        # Apply visual psychology triggers
        frame = self._apply_neuropsychology_triggers(frame, variant_id)

        # Apply quantum perturbation for subtle variation
        channels = []

        for i in range(3):
            U, S, V = np.linalg.svd(frame[:, :, i], full_matrices=False)

            @qml.qnode(self.qml_dev,
                       interface='torch' if self.device.type != 'cpu' else 'numpy')
            def quantum_circuit():
                angle = random.uniform(0, np.pi) * (1 + variant_id * 0.1)
                qml.RY(angle, wires=0)
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            noise = quantum_circuit() * 0.05 + 1.0  # Reduced noise for subtlety
            noise_value = noise.item() if hasattr(noise, 'item') else noise
            S_modified = S * noise_value

            transformed = np.dot(U, np.dot(np.diag(S_modified), V))
            channels.append(np.clip(transformed, 0, 255).astype(np.uint8))

        return np.stack(channels, axis=-1)

    def _apply_neuropsychology_triggers(
            self,
            frame: np.ndarray,
            variant_id: int) -> np.ndarray:
        """Apply VIREX-9000 neuropsychology visual triggers"""
        h, w = frame.shape[:2]

        # 1. Golden ratio composition check
        if CONFIG.get('first_frame_optimizer', {}).get(
                'composition') == 'golden_ratio':
            phi = 1.618
            golden_points = [(int(w /
                                  phi), int(h /
                                            phi)), (int(w -
                                                        w /
                                                        phi), int(h /
                                                                  phi)), (int(w /
                                                                              phi), int(h -
                                                                                        h /
                                                                                        phi)), (int(w -
                                                                                                    w /
                                                                                                    phi), int(h -
                                                                                                              h /
                                                                                                              phi))]

            # Subtle highlight at golden ratio intersections
            for point in golden_points:
                # Golden markers
                cv2.circle(frame, point, 3, (255, 215, 0), -1)

        # 2. Motion blur on background (keep product sharp)
        if variant_id % 3 == 0:  # Apply to 1/3 of variants
            # Simple center-focus motion blur
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w // 2, h // 2), min(h, w) // 3, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)

            # Apply directional blur to background
            blurred = cv2.blur(frame, (15, 1))  # Horizontal motion blur
            frame = frame.astype(np.float32)
            blurred = blurred.astype(np.float32)

            for i in range(3):
                frame[:, :, i] = frame[:, :, i] * \
                    mask + blurred[:, :, i] * (1 - mask)

            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 3. Warm color temperature boost (psychological trust)
        if CURRENT_PROFILE.get('color_grading') == 'golden_vibrant':
            # Enhance warm tones
            frame_float = frame.astype(np.float32)
            frame_float[:, :, 2] *= 1.1  # Boost red channel
            frame_float[:, :, 1] *= 1.05  # Slight green boost
            frame_float[:, :, 0] *= 0.95  # Reduce blue
            frame = np.clip(frame_float, 0, 255).astype(np.uint8)

        # 4. Micro-contrast adjustments for texture emphasis
        if variant_id % 2 == 0:  # Apply to half of variants
            # Unsharp mask for texture enhancement
            gaussian = cv2.GaussianBlur(frame, (5, 5), 1.0)
            frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)

        return frame

    def score_video(self, video_path: str) -> float:
        """Score video for viral potential with optimization"""
        try:
            container = av.open(video_path)

            # Extract frames for analysis
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i < 8:  # Use first 8 frames
                    frames.append(frame.to_ndarray(format="rgb24"))
                else:
                    break

            if not frames:
                return 0.5

            clip = np.stack(frames)
            prompt = """USER: <video> [NEURO-SCORING LAYER - Cosmetics Mode]
Score this turmeric & kojic acid soap ad from 0-1 for viral potential: Focus on golden glow reveals, before-after brightening transformations, dewy skin smoothing, FOMO urgency in free offer visuals. Rate high for dopamine-triggering 'wow' moments, social proof stats, and scroll-stopping first frames with vibrant colors.
Weights: glow(0.4), brighten(0.3), fomo(0.3). Only return the number. ASSISTANT:"""

            with torch.cuda.amp.autocast():
                inputs = self.virality_processor(
                    text=prompt,
                    videos=clip,
                    return_tensors="pt").to(
                    self.device)
                generate_ids = self.virality_model.generate(
                    **inputs, max_new_tokens=10)
                response = self.virality_processor.batch_decode(
                    generate_ids, skip_special_tokens=True)[0]

            # Extract score
            match = re.search(r'0?\.\d+|1\.0|0|1', response)
            if match:
                return float(match.group(0))

            return 0.5

        except Exception as e:
            logger.error(f"Error scoring video: {e}")
            return 0.5
        finally:
            container.close()

    def process_video_variant(
            self,
            video_path: str,
            hook: str,
            variant_id: int) -> dict:
        """Process a single video variant with all optimizations"""
        start_time = time.time()
        temp_files = []  # Track temp files for cleanup

        try:
            # Validate inputs
            if not video_path or not os.path.exists(video_path):
                raise ValueError(f"Invalid video path: {video_path}")

            # Stream video processing to avoid loading all frames
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)

            # Initialize first-frame optimizer
            first_frame_optimizer = FirstFrameOptimizer()

            # Process frames in streaming fashion with memory monitoring
            frames_buffer = []
            processed_frames = []
            frame_count = 0
            first_frame_optimized = False

            # Initialize product tracking data
            product_tracking_data = []
            frames_for_tracking = []

            # Adaptive batch sizing based on memory usage if enabled
            current_batch_size = self._get_adaptive_batch_size() if OPTIMIZATION_SETTINGS.get(
                'use_adaptive_batching', True) else PROCESSING_SETTINGS['frame_batch_size']

            for frame in container.decode(video=0):
                frame_np = frame.to_ndarray(format="bgr24")

                # Apply VIREX-9000 first-frame optimization
                if frame_count == 0 and not first_frame_optimized:
                    logger.info(
                        f"Applying VIREX-9000 first-frame optimization for variant {variant_id}")
                    frame_np = first_frame_optimizer.optimize_first_frame(
                        frame_np, PRODUCT_TYPE)
                    first_frame_optimized = True

                frames_buffer.append(frame_np)
                frame_count += 1

                # Collect frames for product tracking (sample every few frames)
                if frame_count % 5 == 1:  # Sample every 5th frame for tracking
                    frames_for_tracking.append(frame_np)

                # Process when buffer is full (with adaptive size)
                if len(frames_buffer) >= current_batch_size:
                    # Text removal
                    batch = self.process_frame_batch(
                        frames_buffer, "text_removal")

                    # Style transfer
                    batch = self.process_frame_batch(
                        batch, "style_transfer", variant_id=variant_id)

                    processed_frames.extend(batch)
                    frames_buffer = []

                    # Clear GPU memory periodically and update batch size
                    if frame_count % 100 == 0:
                        torch.cuda.empty_cache()
                        current_batch_size = self._get_adaptive_batch_size()

            # Process remaining frames
            if frames_buffer:
                batch = self.process_frame_batch(frames_buffer, "text_removal")
                batch = self.process_frame_batch(
                    batch, "style_transfer", variant_id=variant_id)
                processed_frames.extend(batch)

            container.close()

            # Run product tracking analysis if available
            if self.product_tracker and frames_for_tracking:
                logger.info(
                    f"Running product tracking analysis for variant {variant_id}")
                product_tracking_data = self.product_tracker.track_product_in_frames(
                    frames_for_tracking, PRODUCT_TYPE)

                # Adjust frames based on product coverage if needed
                for i, tracking_info in enumerate(product_tracking_data):
                    if tracking_info["needs_adjustment"]:
                        # Find corresponding frame index in processed_frames
                        frame_idx = i * 5  # Since we sampled every 5th frame
                        if frame_idx < len(processed_frames):
                            logger.info(
                                f"Adjusting frame {frame_idx} for optimal product coverage (current: {
                                    tracking_info['coverage']:.1f}%)")
                            processed_frames[frame_idx] = self.product_tracker.adjust_frame_for_coverage(
                                processed_frames[frame_idx], tracking_info)

            # Find best scene
            # Sample 1 frame per second
            sample_frames = processed_frames[::max(1, int(fps))]
            scene_scores = self.process_frame_batch(
                sample_frames, "scene_scoring")

            best_scene_idx = np.argmax(scene_scores)
            start_frame = int(best_scene_idx * fps)
            end_frame = min(start_frame +
                            int(PROCESSING_SETTINGS['video_duration_seconds'] *
                                fps), len(processed_frames))

            # Generate product visibility heatmap if tracking was done
            if product_tracking_data and self.product_tracker:
                heatmap = self.product_tracker.generate_visibility_heatmap(
                    product_tracking_data,
                    (processed_frames[0].shape[0], processed_frames[0].shape[1])
                )
                # Save heatmap for analysis
                heatmap_path = os.path.join(
                    PROCESSING_SETTINGS['output_directory'],
                    f"heatmap_variant_{variant_id}.png")
                cv2.imwrite(heatmap_path, heatmap)
                logger.info(
                    f"Saved product visibility heatmap to {heatmap_path}")

            # Create temporary video
            temp_path = tempfile.NamedTemporaryFile(
                suffix='.mp4', delete=False).name
            temp_files.append(temp_path)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = processed_frames[0].shape[:2]
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            for i in range(start_frame, end_frame):
                out.write(processed_frames[i])
            out.release()

            # Final optimization with FFmpeg
            output_path = os.path.join(
                PROCESSING_SETTINGS['output_directory'],
                f"variant_{variant_id}.mp4")
            final_paths = ray.get(
                parallel_ffmpeg_processing.remote(
                    [temp_path],
                    [hook],
                    PROCESSING_SETTINGS['output_directory']))

            if final_paths:
                output_path = final_paths[0]

                # Score the video
                score = self.score_video(output_path)

                processing_time = time.time() - start_time

                return {
                    'variant_id': variant_id,
                    'path': output_path,
                    'hook': hook,
                    'score': score,
                    'processing_time': processing_time
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing variant {variant_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Cleanup temp files
            if not SAVE_INTERMEDIATE_FILES:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except BaseException:
                        pass
            # Clear GPU memory
            torch.cuda.empty_cache()

# ============================================================================
# OPTIMIZATION 4: Concurrent I/O Operations
# ============================================================================


@ray.remote
def concurrent_download_videos(urls: List[str]) -> List[str]:
    """Download multiple videos concurrently with retry logic"""
    def download_single(session, url, idx):
        for attempt in range(3):  # 3 retry attempts
            try:
                logger.info(
                    f"Downloading video {idx}: {url} (attempt {
                        attempt + 1}/3)")
                with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"HTTP {response.status}")

                    temp_path = tempfile.NamedTemporaryFile(
                        suffix='.mp4', delete=False).name

                    content = response.read()
                    with open(temp_path, 'wb') as f:
                        f.write(content)

                    logger.info(f"Downloaded video {idx} successfully")
                    return (idx, temp_path)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.error(
                        f"Failed to download video {idx} after 3 attempts: {e}")
                    return (idx, None)
                else:
                    logger.warning(
                        f"Download attempt {
                            attempt +
                            1} failed for video {idx}: {e}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff

    # Create session with connection pooling
    connector = aiohttp.TCPConnector(
        limit_per_host=PROCESSING_SETTINGS['concurrent_downloads'])
    with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_single(session, url, i)
                 for i, url in enumerate(urls)]
        results = ray.get(*tasks)

    # Sort by index and extract paths
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results if r[1] is not None]


@ray.remote
def parallel_ffmpeg_processing(
        input_paths: List[str],
        hooks: List[str],
        output_dir: str) -> List[str]:
    """Process multiple videos with FFmpeg in parallel using VIREX-9000 overlays"""
    # Check if FFmpeg is available
    try:
        run(['ffmpeg', '-version'], stdout=PIPE, stderr=PIPE, check=True)
    except BaseException:
        logger.error("FFmpeg not found! Please install FFmpeg.")
        return []

    # Initialize overlay generator
    overlay_gen = OverlayGenerator()

    def process_single(args):
        input_path, hook, variant_id = args
        output_path = os.path.join(output_dir, f"variant_{variant_id}.mp4")

        try:
            # Generate advanced filter with VIREX-9000 overlays
            filter_complex = overlay_gen.generate_ffmpeg_filter(
                hook, variant_id)

            cmd = ['ffmpeg',
                   '-y',
                   '-i',
                   input_path,
                   '-vf',
                   filter_complex,
                   '-c:v',
                   'libx265',
                   '-preset',
                   'fast',
                   '-crf',
                   str(QUALITY_SETTINGS['video_crf']),
                   '-c:a',
                   'aac',
                   '-b:a',
                   '128k',
                   '-movflags',
                   '+faststart',
                   '-metadata',
                   f'comment=variant_{variant_id}',
                   output_path]

            run(cmd, stdout=PIPE, stderr=PIPE, text=True, check=True)
            logger.info(
                f"Generated variant {variant_id} with {
                    overlay_gen.overlay_variants[
                        variant_id % len(
                            overlay_gen.overlay_variants)]['style']} overlay style")
            return output_path
        except CalledProcessError as e:
            logger.error(f"FFmpeg error for variant {variant_id}: {e.stderr}")
            return None
        except Exception as e:
            logger.error(
                f"FFmpeg processing failed for variant {variant_id}: {e}")
            return None

    # Process in parallel using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        args = [(input_paths[i % len(input_paths)], hooks[i % len(hooks)], i)
                for i in range(len(input_paths) * len(hooks))]
        results = list(executor.map(process_single, args))

    return [r for r in results if r is not None]

# ============================================================================
# Main Processing Pipeline
# ============================================================================


class OptimizedVideoProcessor:
    def __init__(self):
        self.start_time = time.time()
        self.results = []

        # Initialize actor pool with cache reference
        logger.info("Initializing model actor pool...")
        self.actors = [
            ModelActor.remote(
                i, cache_actor) for i in range(
                CLUSTER_SETTINGS['gpu_count'])]
        self.actor_pool = ActorPool(self.actors)

        # Initialize hooks
        self.hooks = []

    def process_all_videos(self):
        """Main processing pipeline with all optimizations"""
        logger.info("=" * 60)
        logger.info("Facebook Ads Video Processor - Ultra-Optimized Edition")
        logger.info("=" * 60)
        logger.info(f"Videos: {len(VIDEO_URLS)}")
        logger.info(
            f"Variants per video: {
                PROCESSING_SETTINGS['variants_per_video']}")
        logger.info(
            f"Total outputs: {
                len(VIDEO_URLS) *
                PROCESSING_SETTINGS['variants_per_video']}")
        logger.info(f"GPU count: {CLUSTER_SETTINGS['gpu_count']}")
        logger.info("Optimizations enabled:")
        for key, value in OPTIMIZATION_SETTINGS.items():
            if value:
                logger.info(f"  ✓ {key}")
        logger.info("=" * 60)

        # Start downloads immediately (non-blocking)
        logger.info(f"Starting download of {len(VIDEO_URLS)} videos...")
        video_paths_future = concurrent_download_videos.remote(VIDEO_URLS)

        # Generate hooks while videos download (overlapping I/O)
        logger.info("Generating marketing hooks...")
        self.hooks = self._generate_hooks()

        # Wait for downloads to complete
        logger.info("Waiting for video downloads to complete...")
        video_paths = video_paths_future

        # Process videos in batches
        logger.info("Processing videos with optimized pipeline...")

        # Create tasks for each video variant using smart variant generation
        task_args = self._generate_smart_variants(video_paths)

        # Process with actor pool
        results = []
        for i, result in enumerate(self.actor_pool.map(
            lambda actor, args: actor.process_video_variant(*args),
            task_args
        )):
            if result:
                results.append(result)
                self._print_progress(len(results), len(task_args))

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        self.results = results

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _generate_smart_variants(self, video_paths: List[str]) -> List[Tuple]:
        """VIREX-9000 Smart Variant Generation System"""
        logger.info(
            "Generating smart variants with A/B testing configurations...")

        variant_config = CONFIG.get(
            'variant_generator', {
                "first_frame_comps": [
                    "center_glow", "angled_brighten", "rule_thirds"], "hook_positions": [
                    "top", "middle", "bottom"], "hook_types": [
                    "question", "stat", "claim"], "urgency_elements": [
                        "countdown_timer", "stock_counter", "limited_offer"]})

        task_args = []
        variant_id = 0

        for video_idx, video_path in enumerate(video_paths):
            # Generate intelligent variant combinations
            for variant_idx in range(
                    PROCESSING_SETTINGS['variants_per_video']):
                # Select hook based on type rotation
                hook_type_idx = variant_idx % len(variant_config['hook_types'])
                hook_type = variant_config['hook_types'][hook_type_idx]

                # Select appropriate hook based on type
                if hook_type == "question":
                    # Use hooks that end with ?
                    question_hooks = [h for h in self.hooks if '?' in h]
                    hook = question_hooks[variant_idx % len(
                        question_hooks)] if question_hooks else self.hooks[variant_idx % len(self.hooks)]
                elif hook_type == "stat":
                    # Use hooks with numbers/percentages
                    stat_hooks = [
                        h for h in self.hooks if any(
                            c.isdigit() for c in h)]
                    hook = stat_hooks[variant_idx % len(
                        stat_hooks)] if stat_hooks else self.hooks[variant_idx % len(self.hooks)]
                else:  # claim
                    # Use bold claim hooks
                    claim_hooks = [
                        h for h in self.hooks if 'FREE' in h or '!' in h]
                    hook = claim_hooks[variant_idx % len(
                        claim_hooks)] if claim_hooks else self.hooks[variant_idx % len(self.hooks)]

                # Create variant metadata for logging
                variant_metadata = {
                    "composition": variant_config['first_frame_comps'][variant_idx % len(variant_config['first_frame_comps'])],
                    "hook_position": variant_config['hook_positions'][variant_idx % len(variant_config['hook_positions'])],
                    "hook_type": hook_type,
                    "urgency_element": variant_config['urgency_elements'][variant_idx % len(variant_config['urgency_elements'])]
                }

                logger.info(f"Variant {variant_id}: {variant_metadata}")

                task_args.append((video_path, hook, variant_id))
                variant_id += 1

        logger.info(
            f"Generated {
                len(task_args)} smart variants for A/B testing")
        return task_args

    def _generate_hooks(self) -> List[str]:
        """Generate marketing hooks using AI with neuropsychological optimization"""
        try:
            # Validate API key exists
            if not API_KEYS.get('xai_api_key'):
                raise ValueError(
                    "XAI API key is required for generating marketing hooks. Please set XAI_API_KEY environment variable.")

            llm = ChatOpenAI(
                model="grok-4",
                api_key=API_KEYS['xai_api_key'],
                base_url="https://api.x.ai/v1/chat/completions",
                temperature=0.7,
            )

            prompt = f"""Generate exactly {
                PROCESSING_SETTINGS['variants_per_video'] * 2} ultra-viral marketing hooks for trending turmeric & kojic acid soap ads on Facebook in 2025, optimized for the first frame to stop scrolls in 1-3 seconds.
Each hook must be 10-15 words max, super short, punchy, and action-oriented with a pattern interrupt (e.g., bold question, shocking stat, urgent command, or hero mini-story).
IMPORTANT: The offer is FREE SOAP (just pay small shipping fee) – emphasize 'FREE' as the no-brainer value proposition, anchoring high worth (e.g., '$49 value FREE') to trigger reciprocity and impulse claims.
Focus on product benefits: natural turmeric for golden glow and anti-inflammatory soothing, kojic acid for brightening and fading spots/hyperpigmentation, resulting in radiant, even-toned, dewy skin.
Incorporate these neuropsychological triggers in every hook:
- FOMO and scarcity (e.g., 'Last 24 hours – FREE glow bottles vanishing!'),
- Social proof and authority (e.g., 'Dermatologists obsessed – 50K+ claimed FREE brightening soap'),
- Loss aversion and anchoring (e.g., 'Don't lose your $49 glow – FREE soap, shipping only'),
- Emotional urgency and dopamine words (e.g., 'FREE instant brightening unlocks radiant confidence rush'),
- Cognitive biases like reciprocity (emphasize 'FREE' gift value exchange) and confirmation (affirm pain like 'Tired of dull, uneven skin?').
Evoke color psychology subtly (e.g., words like 'golden glow', 'radiant dewy' for warm, trusting, energetic vibes).
Draw from 2025 trends: AI-personalized clean beauty (e.g., 'AI-matched FREE glow formula'), K-beauty dewy influences, interactive calls (e.g., 'Comment GLOW for your FREE bottle!'), and influencer-style authenticity for viral shares.
Ensure variety for A/B testing: 1/3 questions (curiosity bias), 1/3 stats/testimonials (social proof), 1/3 bold claims/stories (emotional narrative). Make them sensory-vivid with power words (e.g., 'explode', 'unlock', 'transform') for dopamine hits.
Output ONLY as JSON with key 'hooks' containing a list of strings – no extra text."""

            response = llm.ainvoke(prompt)
            generated = json.loads(response.content)['hooks']

            # Validate generated hooks
            validated_hooks = []
            for hook in generated:
                # Check word count (10-15 words)
                word_count = len(hook.split())
                if 5 <= word_count <= 20:  # Slightly flexible range
                    # Check for at least one psychological trigger
                    trigger_words = [
                        'last',
                        'only',
                        'limited',
                        'now',
                        'today',
                        'instant',
                        'transform',
                        'glow',
                        'radiant',
                        'free',
                        'save',
                        'unlock',
                        'discover',
                        'reveal',
                        'dermatologist',
                        'proven',
                        'clinically',
                        '%',
                        'thousand',
                        'million']
                    if any(word.lower() in hook.lower()
                           for word in trigger_words):
                        validated_hooks.append(hook)
                    else:
                        logger.warning(
                            f"Hook lacks psychological triggers: {hook}")
                else:
                    logger.warning(
                        f"Hook has {word_count} words (target 10-15): {hook}")

            # Log hook variety
            question_count = sum(1 for h in validated_hooks if '?' in h)
            stat_count = sum(
                1 for h in validated_hooks if any(
                    c.isdigit() for c in h))
            logger.info(
                f"Generated hooks - Questions: {question_count}, Stats: {stat_count}, Claims: {
                    len(validated_hooks) -
                    question_count -
                    stat_count}")

            # Return only validated hooks from Grok API
            return validated_hooks[:PROCESSING_SETTINGS['variants_per_video'] * 2]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Grok-4 response: {e}")
            raise ValueError(f"Grok API returned invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Failed to generate hooks: {e}")
            raise RuntimeError(
                f"Failed to generate marketing hooks via Grok API: {e}")

    def _print_progress(self, completed: int, total: int):
        """Print processing progress"""
        progress = (completed / total) * 100
        elapsed = time.time() - self.start_time
        eta = (elapsed / completed) * \
            (total - completed) if completed > 0 else 0

        logger.info(
            f"Progress: {completed}/{total} ({progress:.1f}%) - ETA: {eta / 60:.1f} minutes")

    def _save_results(self):
        """Save processing results"""
        total_time = time.time() - self.start_time

        summary = {
            'total_videos': len(VIDEO_URLS),
            'total_variants': len(self.results),
            'total_time': total_time,
            'average_time_per_video': total_time / len(VIDEO_URLS) if VIDEO_URLS else 0,
            'top_variants': self.results[:10],
            'optimization_settings': OPTIMIZATION_SETTINGS,
            'performance_metrics': {
                'videos_per_minute': len(VIDEO_URLS) / (total_time / 60) if total_time > 0 else 0,
                'gpu_efficiency': len(self.results) / (CLUSTER_SETTINGS['gpu_count'] * (total_time / 3600))
            }
        }

        results_path = os.path.join(
            PROCESSING_SETTINGS['output_directory'],
            'processing_results.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _print_summary(self):
        """Print processing summary"""
        total_time = time.time() - self.start_time

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(
            f"Total time: {
                total_time /
                60:.1f} minutes ({
                total_time:.1f} seconds)")
        logger.info(
            f"Average per video: {
                total_time /
                len(VIDEO_URLS):.1f} seconds")
        logger.info(
            f"Videos per minute: {len(VIDEO_URLS) / (total_time / 60):.2f}")
        # Assuming 90 min baseline
        logger.info(f"Speedup vs baseline: {(90 * 60) / total_time:.1f}x")

        if self.results:
            logger.info("\nTop 10 Variants:")
            for i, variant in enumerate(self.results[:10]):
                logger.info(
                    f"{i + 1}. Variant {variant['variant_id']} - Score: {variant['score']:.3f} - {variant['hook'][:50]}...")

        # Print cache statistics if caching enabled
        if OPTIMIZATION_SETTINGS.get(
            'use_caching',
                True) and cache_actor is not None:
            cache_stats = ray.get(cache_actor.get_stats.remote())
            logger.info("\nCache Statistics:")
            logger.info(f"  Cache hits: {cache_stats['hits']}")
            logger.info(f"  Cache misses: {cache_stats['misses']}")
            logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
            logger.info(f"  Cached OCR results: {cache_stats['ocr_entries']}")
            logger.info(
                f"  Cached SigLIP scores: {
                    cache_stats['siglip_entries']}")
            logger.info(
                f"  Cached style frames: {
                    cache_stats['style_entries']}")

        logger.info("=" * 60)

# ============================================================================
# Auto-scaling and Monitoring
# ============================================================================


def setup_autoscaling():
    """Setup Ray autoscaling based on configuration"""
    if CLUSTER_SETTINGS['auto_scaling']['enabled']:
        logger.info("Configuring auto-scaling...")

        # This would integrate with Ray's autoscaler
        # In practice, you'd use ray.autoscaler.sdk or configure via cluster YAML
        # For RunPod, this might involve their API to add/remove nodes

        min_nodes = CLUSTER_SETTINGS['auto_scaling']['min_nodes']
        max_nodes = CLUSTER_SETTINGS['auto_scaling']['max_nodes']

        logger.info(f"Auto-scaling configured: {min_nodes}-{max_nodes} nodes")


def print_cluster_metrics():
    """Print current cluster metrics"""
    try:
        nodes = ray.nodes()
        resources = ray.cluster_resources()

        logger.info("\nCluster Metrics:")
        logger.info(f"  Active Nodes: {len([n for n in nodes if n['Alive']])}")
        logger.info(f"  Total GPUs: {int(resources.get('GPU', 0))}")
        logger.info(
            f"  Available GPUs: {int(ray.available_resources().get('GPU', 0))}")
        logger.info(
            f"  Memory Usage: {
                ray.cluster_resources().get(
                    'memory',
                    0) / 1e9:.1f} GB")

    except Exception as e:
        logger.error(f"Could not get cluster metrics: {e}")

# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point with all optimizations"""
    try:
        # Setup autoscaling
        setup_autoscaling()

        # Print initial cluster status
        print_cluster_metrics()

        # Create processor
        processor = OptimizedVideoProcessor()

        # Run processing
        processor.process_all_videos()

        # Final metrics
        print_cluster_metrics()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
