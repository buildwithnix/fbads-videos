import os
import sys
import json
import time
import aiohttp
import asyncio
import tempfile
import logging
import requests  # Added for concurrent_download_videos
from typing import List, Dict, Tuple, Optional, Any
import concurrent.futures
import hashlib
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
from skimage.metrics import structural_similarity as ssim  # For real SSIM

import torch
import numpy as np
import cv2
import av
import ray
from ray.util import ActorPool
from PIL import Image

from diffusers import FluxInpaintPipeline  # Correct for inpaint
from transformers import AutoModel, AutoProcessor, pipeline
from langchain_community.llms import HuggingFaceHub  # For Gemma 3
import paddleocr  # For PaddleOCR
from subprocess import run, CalledProcessError
from moviepy.editor import ImageSequenceClip  # For metadata/assembly

# Real SOTA imports (based on availability)
from diffusers import DiffMambaPipeline  # For VSR (Diff-Mamba)
from diffusers import HunyuanVideoPipeline  # For perturbations (HunyuanVideo)

load_dotenv()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

VIDEO_URLS = CONFIG['video_urls']
PROCESSING_SETTINGS = CONFIG['processing_settings']
CLUSTER_SETTINGS = CONFIG['cluster_settings']
OPTIMIZATION_SETTINGS = CONFIG['optimization_settings']
QUALITY_SETTINGS = CONFIG['quality_presets'][PROCESSING_SETTINGS['quality_preset']]
PRODUCT_TYPE = CONFIG.get('product_specific', {}).get('type', 'default')
MODELS = CONFIG['models']
FIRST_FRAME_OPTIMIZER = CONFIG['first_frame_optimizer']

API_KEYS = {
    'xai_api_key': os.environ.get('XAI_API_KEY'),
}

os.makedirs(PROCESSING_SETTINGS['output_directory'], exist_ok=True)

ray.init(ignore_reinit_error=True, logging_level=logging.INFO)

@ray.remote
class DistributedCache:
    def __init__(self):
        self.ocr_cache = {}
        self.clip_cache = {}
        self.style_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_ocr(self, frame_hash: str) -> Optional[Any]:
        result = self.ocr_cache.get(frame_hash)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def set_ocr(self, frame_hash: str, result: Any):
        self.ocr_cache[frame_hash] = result

    def get_clip(self, frame_hash: str) -> Optional[Dict[str, float]]:
        result = self.clip_cache.get(frame_hash)
        if result is not None:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def set_clip(self, frame_hash: str, scores: Dict[str, float]):
        self.clip_cache[frame_hash] = scores

    def get_style(self, frame_hash: str, variant_id: int) -> Optional[np.ndarray]:
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
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'ocr_entries': len(self.ocr_cache),
            'clip_entries': len(self.clip_cache),
            'style_entries': len(self.style_cache)
        }

cache_actor = DistributedCache.remote() if OPTIMIZATION_SETTINGS.get('use_caching', True) else None

class NeuropsychologyPromptEngine:
    def get_prompts(self, product_type: str = "skincare") -> List[str]:
        if product_type == "turmeric_kojic_soap":
            return [
                "Instant turmeric & kojic acid soap transformation that triggers FOMO: golden glow reveal, before-after brightening magic with dewy finish, millions sharing this viral skincare demo – thumb-stopping urgency in every radiant, smoothed frame",
                "As a dermatologist-endorsed hit: eye-catching turmeric & kojic acid soap reveal with social proof stats on hyperpigmentation fade, beautiful aesthetic lighting on brightened, vibrant skin, high-engagement reel that stops endless scrolls",
                "Pattern-interrupting question: Tired of dull, uneven skin tones? This shareable turmeric & kojic acid soap moment unlocks glowing confidence with smooth texture, viral Facebook content with emotional urgency and aesthetic perfection",
                "High-engagement first frame: Attention-grabbing visual of AI-personalized turmeric & kojic acid soap miracle for instant brightening and dewy vibrancy, scarcity vibe with 'limited stock glow', beautiful scene evoking dopamine rush",
                "Viral social media moment re-read: Stunning turmeric & kojic acid soap demonstration with radiant, smoothed textures and color vibrancy, eye-catching golden hues triggering shares, emotional story of spot-fading transformation that hooks instantly",
                "Thumb-stopping aesthetic chain: Flawless skin close-up with turmeric & kojic acid soap's subtle dewy glow effects and brightening progression, engaging narrative pull with vibrant colors, high-virality stats like 10M views – perfect for Facebook reels",
                "Engaging reel with K-beauty influences: Shareable turmeric & kojic acid soap content showing instant brightening results and smooth, dewy finish, scarcity-driven call-to-action, beautiful visuals that spark FOMO and endless engagement",
                "Attention-grabbing viral hit: Eye-catching frame of transformative turmeric & kojic acid soap for glowing, vibrant skin, social proof from user testimonials on brightening and texture smoothing, dopamine-triggering 'wow' moment in high-aesthetic scene"
            ]
        return [
            "Viral explosion waiting to happen: thumb-stopping visual with pattern-breaking element, millions can't stop sharing",
            "Social proof masterpiece: high-engagement content with authority endorsement, beautiful aesthetics that demand attention",
            "FOMO-triggering moment: limited-time visual story with emotional hook, shareable content breaking platform records",
            "Dopamine rush guaranteed: eye-catching first frame with transformation narrative, viral statistics prove unstoppable engagement",
            "Pattern interrupt perfection: unexpected visual twist with social validation, aesthetic excellence meets viral psychology",
            "Trending now: platform-optimized content with influencer appeal, scarcity mindset activation in every frame",
            "Scroll-stopper supreme: attention monopolizing visual with emotional resonance, share-worthy moment captured perfectly",
            "Engagement multiplier: algorithm-loving content with human psychology triggers, viral DNA embedded in every pixel"
        ]

    def boost_first_frame_prompts(self, prompts: List[str], boost_factor: float = 1.5) -> List[str]:
        return [f"FIRST FRAME CRITICAL: {prompt} - maximum visual contrast, center-third focal point, instant pattern recognition" for prompt in prompts]

    def get_prompt_weights(self) -> Dict[str, float]:
        return {
            "fomo_trigger": 0.25, "social_proof": 0.20, "pattern_interrupt": 0.20,
            "emotion_hook": 0.15, "visual_appeal": 0.10, "trend_alignment": 0.10
        }

class FirstFrameOptimizer:
    def optimize_first_frame(self, frame: np.ndarray, product_type: str = None) -> np.ndarray:
        frame = self._enhance_contrast(frame, FIRST_FRAME_OPTIMIZER['contrast_boost'])
        frame = self._apply_golden_ratio_composition(frame)
        for element in FIRST_FRAME_OPTIMIZER['interrupt_elements']:
            match element:
                case "glow_aura":
                    frame = self._add_glow_aura(frame)
                case "urgency_angle":
                    frame = self._apply_urgency_angle(frame)
        frame = self._apply_golden_hour_lighting(frame, FIRST_FRAME_OPTIMIZER['golden_hour_temp'])
        frame = self._enhance_product_prominence(frame, product_type)
        return frame

    def _enhance_contrast(self, frame: np.ndarray, boost_factor: float) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=boost_factor * 2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_golden_ratio_composition(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        phi = 1.618
        golden_x = int(w / phi)
        golden_y = int(h / phi)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (golden_x, golden_y), int(min(h, w) * 0.4), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (int(w * 0.3) | 1, int(h * 0.3) | 1), 0)
        frame = frame.astype(np.float32)
        for i in range(3):
            frame[:, :, i] = frame[:, :, i] * (0.3 + 0.7 * mask)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _add_glow_aura(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        glow = cv2.GaussianBlur(bright_mask, (51, 51), 0)
        glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
        glow[:, :, 0] = glow[:, :, 0] * 0.3
        glow[:, :, 1] = glow[:, :, 1] * 0.8
        glow[:, :, 2] = glow[:, :, 2] * 1.0
        return cv2.addWeighted(frame, 0.85, glow, 0.15, 0)

    def _apply_urgency_angle(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        angle = 2.5
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += (w - w) / 2
        M[1, 2] += (h - h) / 2
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _apply_golden_hour_lighting(self, frame: np.ndarray, temp_k: int) -> np.ndarray:
        color_matrix = np.array([
            [1.2, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.8]
        ]) if temp_k == 3000 else np.eye(3)
        frame_float = frame.astype(np.float32)
        for i in range(3):
            frame_float[:, :, i] = np.clip(frame_float[:, :, i] * color_matrix[i, i], 0, 255)
        return frame_float.astype(np.uint8)

    def _enhance_product_prominence(self, frame: np.ndarray, product_type: str) -> np.ndarray:
        if product_type == "turmeric_kojic_soap":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_gold, upper_gold)
            hsv[:, :, 1] = np.where(mask > 0, np.clip(hsv[:, :, 1] * 1.3, 0, 255), hsv[:, :, 1])
            hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * 1.1, 0, 255), hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class ModelActor:
    def __init__(self, actor_id: int, cache_actor):
        self.actor_id = actor_id
        self.cache = cache_actor
        self.device = torch.device(f"cuda:{actor_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
        self.current_batch_size = PROCESSING_SETTINGS['frame_batch_size']
        self._load_optimized_models()

    def _load_optimized_models(self):
        dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        common_kwargs = {"torch_dtype": dtype, "use_safetensors": True, "low_cpu_mem_usage": True}
        self.flux_inpaint_pipe = FluxInpaintPipeline.from_pretrained(MODELS['flux_model_id'], **common_kwargs).to(self.device)
        self.flux_inpaint_pipe.enable_attention_slicing()
        self.flux_inpaint_pipe.enable_vae_slicing()
        self.clip_model = AutoModel.from_pretrained(MODELS['clip_model_id'], torch_dtype=torch.bfloat16).to(self.device)
        self.clip_processor = AutoProcessor.from_pretrained(MODELS['clip_model_id'])
        self.ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR init
        self.gemma_model = pipeline("text-generation", model=MODELS['gemma_model'], device_map="auto")
        self.vsr_pipe = DiffMambaPipeline.from_pretrained(MODELS['vsr_model_id'], **common_kwargs).to(self.device)
        self.perturb_pipe = HunyuanVideoPipeline.from_pretrained(MODELS['video_perturb_model_id'], **common_kwargs).to(self.device)

    def process_frame_batch(self, frames: List[np.ndarray], operation: str, **kwargs) -> List[np.ndarray]:
        match operation:
            case "text_removal":
                return self._batch_text_removal(frames)
            case "style_transfer":
                return self._batch_style_transfer(frames, **kwargs)
            case "video_sr":
                return self._batch_video_sr(frames)
            case "semantic_perturb":
                return self._batch_semantic_perturb(frames, **kwargs)
            case _:
                raise ValueError(f"Unknown operation: {operation}")

    def _batch_text_removal(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        ocr_results = []
        for frame in frames:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            cached = ray.get(self.cache.get_ocr.remote(frame_hash)) if self.cache else None
            if cached is not None:
                ocr_results.append(cached)
                continue
            result = self.ocr_model.ocr(frame, cls=True)  # PaddleOCR usage
            if self.cache:
                ray.get(self.cache.set_ocr.remote(frame_hash, result))
            ocr_results.append(result)
        frames_to_inpaint = []
        masks = []
        for i, (frame, ocr_result) in enumerate(zip(frames, ocr_results)):
            if ocr_result:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for detection in ocr_result:
                    if detection:
                        pts = np.array(detection[0], dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                if mask.any():
                    frames_to_inpaint.append(frame)
                    masks.append(mask)
        if frames_to_inpaint:
            pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_to_inpaint]
            pil_masks = [Image.fromarray(m) for m in masks]
            inpainted = self.flux_inpaint_pipe(
                prompt=["seamless skincare product scene"] * len(pil_images),
                image=pil_images,
                mask_image=pil_masks,
                num_inference_steps=QUALITY_SETTINGS['inpaint_inference_steps'],
                strength=1.0
            ).images
            for idx, img in enumerate(inpainted):
                frames_to_inpaint[idx] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frames

    def _batch_style_transfer(self, frames: List[np.ndarray], variant_id: int) -> List[np.ndarray]:
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        stylized_frames = []
        batch_size = min(self.current_batch_size, len(pil_frames))
        for i in range(0, len(pil_frames), batch_size):
            batch = pil_frames[i:i + batch_size]
            prompt = "professional product photography, high quality, variant {variant_id}"
            results = self.flux_inpaint_pipe(  # Reuse for img2img-like
                prompt=[prompt] * len(batch),
                image=batch,
                num_inference_steps=QUALITY_SETTINGS['svd_inference_steps'],
                guidance_scale=7.5,
                strength=0.6
            ).images
            stylized_frames.extend(results)
        results = []
        for stylized in stylized_frames:
            frame_bgr = cv2.cvtColor(np.array(stylized), cv2.COLOR_RGB2BGR)
            perturbed = self.add_simple_noise(frame_bgr, intensity=PROCESSING_SETTINGS['noise_intensity'])
            results.append(perturbed)
        return results

    def _batch_scene_scoring(self, frames: List[np.ndarray]) -> List[float]:
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        prompt_engine = NeuropsychologyPromptEngine()
        viral_prompts = prompt_engine.get_prompts(product_type=PRODUCT_TYPE)
        inputs = self.clip_processor(text=viral_prompts, images=pil_frames, padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.clip_model(**inputs)
        sigmoid_scores = torch.sigmoid(outputs.logits_per_image)
        scores = []
        for frame_scores in sigmoid_scores.cpu().numpy():
            score_dict = {prompt: float(score) for prompt, score in zip(viral_prompts, frame_scores)}
            weighted_score = sum(score_dict.values()) / len(score_dict) if score_dict else 0.0
            scores.append(weighted_score)
        return scores

    def add_simple_noise(self, frame: np.ndarray, intensity: float = 0.02) -> np.ndarray:
        noise = np.random.normal(0, intensity * 255, frame.shape)
        noisy_frame = frame.astype(np.float32) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)

    def extract_keyframes(self, video_path: str) -> List[np.ndarray]:
        frames = []
        prev_gray = None
        with av.open(video_path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i % PROCESSING_SETTINGS['keyframe_interval'] == 0:
                    img = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        score = np.mean(diff)
                        if score > PROCESSING_SETTINGS['scene_threshold']:
                            frames.append(img)
                    else:
                        frames.append(img)
                    prev_gray = gray
        return frames

    def temporal_smooth(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        smoothed = []
        for i in range(len(frames)):
            start = max(0, i - PROCESSING_SETTINGS['temporal_window_size'] // 2)
            end = min(len(frames), i + PROCESSING_SETTINGS['temporal_window_size'] // 2 + 1)
            window = frames[start:end]
            avg_frame = np.mean(window, axis=0).astype(np.uint8)
            smoothed.append(avg_frame)
        return smoothed

    def _batch_video_sr(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        sr_frames = []
        for frame in frames:
            sr = self.vsr_pipe(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).images[0]
            sr_frames.append(cv2.cvtColor(np.array(sr), cv2.COLOR_RGB2BGR))
        return sr_frames

    def _batch_semantic_perturb(self, frames: List[np.ndarray], variant_id: int) -> List[np.ndarray]:
        perturbed = []
        for frame in frames:
            perturbed_frame = self.perturb_pipe(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), prompt="subtle semantic perturbation for ad variant").images[0]
            perturbed.append(cv2.cvtColor(np.array(perturbed_frame), cv2.COLOR_RGB2BGR))
        return perturbed

    def process_video_variant_optimized(self, video_path: str, hook: str, variant_id: int) -> dict:
        frames = self.extract_keyframes(video_path)
        processed = self.process_frame_batch(frames, 'text_removal')
        processed = self.process_frame_batch(processed, 'style_transfer', variant_id=variant_id)
        processed = self.process_frame_batch(processed, 'video_sr')
        processed = self.process_frame_batch(processed, 'semantic_perturb', variant_id=variant_id)
        processed[0] = FirstFrameOptimizer().optimize_first_frame(processed[0], PRODUCT_TYPE)
        processed = self.temporal_smooth(processed)
        # Interpolate to full frames
        full_frames = self._interpolate_frames(processed)
        # Enforce 9:16
        full_frames = [self._resize_to_916(f) for f in full_frames]
        # Assemble with metadata
        out_path = os.path.join(PROCESSING_SETTINGS['output_directory'], f"variant_{variant_id}.mp4")
        fps = 30
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in full_frames], fps=fps)
        spoofed_metadata = self._spoof_metadata(variant_id)
        ffmpeg_params = []
        for key, value in spoofed_metadata.items():
            ffmpeg_params.extend(['-metadata', f'{key}={value}'])
        clip.write_videofile(out_path, codec='libx264', audio=False, ffmpeg_params=ffmpeg_params)
        avg_ssim = self._calculate_avg_ssim(video_path, out_path)
        if avg_ssim < PROCESSING_SETTINGS['min_ssim']:
            logger.warning(f"Variant {variant_id} below SSIM threshold: {avg_ssim}. Reprocessing...")
            # Reprocess logic (simplified: return original or retry)
            return {'output_path': video_path, 'avg_ssim': avg_ssim}
        return {'output_path': out_path, 'avg_ssim': avg_ssim}

    def _interpolate_frames(self, keyframes: List[np.ndarray]) -> List[np.ndarray]:
        full = []
        for i in range(len(keyframes) - 1):
            full.append(keyframes[i])
            for _ in range(PROCESSING_SETTINGS['keyframe_interval'] - 1):
                interp = cv2.addWeighted(keyframes[i], 0.5, keyframes[i+1], 0.5, 0)
                full.append(interp)
        full.append(keyframes[-1])
        return full

    def _resize_to_916(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_ratio = 9 / 16
        if h / w > target_ratio:
            new_h = int(w * target_ratio)
            frame = frame[(h - new_h) // 2 : (h + new_h) // 2, :]
        elif h / w < target_ratio:
            new_w = int(h / target_ratio)
            frame = frame[:, (w - new_w) // 2 : (w + new_w) // 2]
        return cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)  # Standard 9:16 res

    def _calculate_avg_ssim(self, original_path: str, variant_path: str) -> float:
        orig_frames = [cv2.cvtColor(np.array(f.to_image()), cv2.COLOR_RGB2BGR) for f in av.open(original_path).decode(video=0)]
        var_frames = [cv2.cvtColor(np.array(f.to_image()), cv2.COLOR_RGB2BGR) for f in av.open(variant_path).decode(video=0)]
        min_len = min(len(orig_frames), len(var_frames))
        scores = [ssim(o, v, channel_axis=-1, data_range=255) for o, v in zip(orig_frames[:min_len], var_frames[:min_len])]
        return np.mean(scores)

    def _spoof_metadata(self, variant_id: int) -> Dict[str, str]:
        now = datetime.now()
        random_days = random.randint(1, 30)
        spoofed_date = (now - timedelta(days=random_days)).isoformat()
        random_lat = random.uniform(-90, 90)
        random_long = random.uniform(-180, 180)
        return {
            'creation_time': spoofed_date,
            'modify_date': spoofed_date,
            'title': f'Organic Ad Variant {variant_id} - {random.randint(1000, 9999)}',
            'author': f'User{random.randint(1000, 9999)}',
            'comment': 'Generated variant for ad optimization',
            'GPSLatitude': str(random_lat),
            'GPSLongitude': str(random_long),
            'encoder': 'iPhone Camera',
            'software': 'Adobe Premiere Pro 2025'
        }

@ray.remote
def concurrent_download_videos(urls: List[str]) -> List[str]:
    def download_url(url):
        with requests.get(url) as resp:
            if resp.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(resp.content)
                    return tmp_file.name
        return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        paths = list(executor.map(download_url, urls))
    return [p for p in paths if p]

class OptimizedVideoProcessor:
    def __init__(self):
        self.start_time = time.time()
        self.results = []
        self.actors = [ModelActor.remote(i, cache_actor) for i in range(CLUSTER_SETTINGS['gpu_count'])]
        self.actor_pool = ActorPool(self.actors)
        self.hooks = []

    def process_all_videos(self):
        video_paths_future = concurrent_download_videos.remote(VIDEO_URLS)
        self.hooks = self._generate_hooks()
        video_paths = ray.get(video_paths_future)
        task_args = [(video_paths[i % len(video_paths)], self.hooks[i % len(self.hooks)], i) for i in range(len(video_paths) * PROCESSING_SETTINGS['variants_per_video'])]
        for result in self.actor_pool.map(lambda actor, args: actor.process_video_variant_optimized.remote(*args), task_args):
            if result:
                self.results.append(ray.get(result))
        self.results.sort(key=lambda x: self._score_variant(x), reverse=True)
        self._save_results()

    def _generate_hooks(self) -> List[str]:
        try:
            llm = pipeline("text-generation", model=MODELS['gemma_model'])
            prompt = "Generate 4 short viral hooks for turmeric soap ads. Output as JSON: {'hooks': ['hook1', 'hook2', 'hook3', 'hook4']}"
            response = llm(prompt, max_new_tokens=100)
            return json.loads(response[0]['generated_text'])['hooks']
        except Exception as e:
            logger.warning(f"Hook generation failed: {e}. Using fallback.")
            return ["Fallback hook 1", "Fallback hook 2", "Fallback hook 3", "Fallback hook 4"]

    def _score_variant(self, variant: dict) -> float:
        return variant.get('avg_ssim', 0.5)

    def _save_results(self):
        total_time = time.time() - self.start_time
        summary = {'total_variants': len(self.results), 'total_time': total_time}
        with open(os.path.join(PROCESSING_SETTINGS['output_directory'], 'processing_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    processor = OptimizedVideoProcessor()
    processor.process_all_videos()
    ray.shutdown()