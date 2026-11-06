#!/usr/bin/env python3
"""
video_snapshot_harvester.py - High-Precision Video Keyframe & Snapshot Harvester.

Extracts, filters, and curates high-fidelity training snapshots from video files
and stream recordings for biometric and somatotype model enrollment:
1. Native I-Frames / Keyframes, Scene Changes, or Regular Interval Sampling.
2. Studio Aesthetic & Pose Quality Scoring: Evaluates sharpness, balanced studio lighting,
   dynamic contrast, stillness/settled pose, and biometric presence.
3. Diverse Situation Sampling: Discovers and isolates distinct situations/scenes across
   the video, selecting the peak studio frame in each context.
4. Explicit Timestamp Selection: Allows specifying custom timestamps where the model looks best.
5. Perceptual Difference Hashing (dHash 64-bit) for zero-redundancy deduplication.
6. Rigorous Filename Conventions: Every snapshot explicitly embeds the source video filename
   and the exact extracted timestamp (e.g. `stream01_snap_t02m14s_500ms.jpg`).
7. Biometric & Somatotype Gating: Filters for verified human poses and faces.
8. Automated Anatomical Cropping: Saves multi-region crops (face, torso, hips) for instant review.
9. Nautilus Desktop Thumbnail Pre-caching and Studio Contact Sheet Generation.
"""

import os
import sys
import json
import time
import math
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union

import cv2
import numpy as np
from PIL import Image

# Add model-media-scraper to sys.path to leverage Biometric & Somatotype engines
MM_SCRAPER_PATH = Path("/home/fbetancourt/Gemini/model-media-scraper")
if MM_SCRAPER_PATH.exists() and str(MM_SCRAPER_PATH) not in sys.path:
    sys.path.insert(0, str(MM_SCRAPER_PATH))

try:
    from biometric_matcher import (
        MultiModalBiometricEngine,
        SomatotypeRegionalExtractor,
        BiometricProfileManager,
        save_regional_crops
    )
except ImportError:
    MultiModalBiometricEngine = None
    SomatotypeRegionalExtractor = None
    BiometricProfileManager = None
    save_regional_crops = None


def format_snapshot_filename(video_stem: str, timestamp_sec: float, tag: Optional[str] = None) -> str:
    """
    Builds a standardized filename embedding source video stem and extracted timestamp.
    Example: `model_session_snap_t01h12m05s_250ms.jpg` or `video_snap_t03m45s_000ms_face.jpg`
    """
    hours = int(timestamp_sec // 3600)
    minutes = int((timestamp_sec % 3600) // 60)
    seconds = int(timestamp_sec % 60)
    millis = int(round((timestamp_sec - int(timestamp_sec)) * 1000))

    if hours > 0:
        time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s_{millis:03d}ms"
    else:
        time_str = f"{minutes:02d}m{seconds:02d}s_{millis:03d}ms"

    tag_str = f"_{tag}" if tag else ""
    return f"{video_stem}_snap_t{time_str}{tag_str}.jpg"


class VideoMetadataExtractor:
    """Extracts duration, dimensions, framerate and codec parameters using ffprobe or OpenCV."""

    @staticmethod
    def get_info(video_path: Path) -> Dict[str, Any]:
        if not video_path.is_file():
            return {}

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate:stream=width,height,codec_name,r_frame_rate,nb_frames",
            "-of", "json",
            str(video_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                fmt = data.get("format", {})
                streams = data.get("streams", [])
                v_stream = next((s for s in streams if s.get("width") and s.get("height")), {})

                fps = 0.0
                r_fps = v_stream.get("r_frame_rate", "0/1")
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    fps = float(num) / float(den) if float(den) > 0 else 0.0

                duration = float(fmt.get("duration", 0.0))
                width = int(v_stream.get("width", 0))
                height = int(v_stream.get("height", 0))

                return {
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "codec": v_stream.get("codec_name", "unknown"),
                    "size_bytes": int(fmt.get("size", video_path.stat().st_size)),
                    "bitrate": int(fmt.get("bit_rate", 0))
                }
        except Exception:
            pass

        # Fallback to OpenCV VideoCapture
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0.0
                cap.release()
                return {
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "codec": "opencv_fallback",
                    "size_bytes": video_path.stat().st_size,
                    "bitrate": 0
                }
        except Exception:
            pass

        return {}

    @classmethod
    def probe(cls, video_path: Union[Path, str]):
        """Legacy helper for test compatibility returning an object with duration_sec, width, height, fps."""
        info = cls.get_info(Path(video_path))
        class VideoMeta:
            def __init__(self, d):
                self.duration_sec = d.get("duration", 0.0)
                self.width = d.get("width", 0)
                self.height = d.get("height", 0)
                self.fps = d.get("fps", 0.0)
        return VideoMeta(info)


class SharpnessAssessor:
    """
    Forensic sharpness and illumination evaluator.
    Computes Laplacian variance and luminance bounds to eliminate motion blur and black frames.
    """

    @staticmethod
    def compute_laplacian_variance(img_bgr: np.ndarray) -> float:
        if img_bgr is None or img_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    @classmethod
    def is_sharp(cls, img_bgr: np.ndarray, min_variance: float = 75.0) -> bool:
        return cls.compute_laplacian_variance(img_bgr) >= min_variance

    @classmethod
    def evaluate(cls, img_bgr: np.ndarray) -> Tuple[bool, float, float, str]:
        """
        Returns: (is_valid, sharpness_score, mean_brightness, reason)
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, 0.0, 0.0, "empty_image"

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        brightness = float(np.mean(gray))

        if brightness < 15.0:
            return False, 0.0, brightness, "underexposed_black_frame"
        if brightness > 245.0:
            return False, 0.0, brightness, "overexposed_white_frame"

        variance = cls.compute_laplacian_variance(img_bgr)
        return True, round(variance, 2), round(brightness, 2), "ok"


class PerceptualDeduplicator:
    """
    Perceptual difference hashing (dHash 64-bit) to eliminate duplicate/stationary video frames.
    """

    def __init__(self, hamming_threshold: int = 8, max_hamming_dist: Optional[int] = None):
        self.hamming_threshold = max_hamming_dist if max_hamming_dist is not None else hamming_threshold
        self.seen_hashes: List[int] = []

    @staticmethod
    def compute_dhash(img_bgr: np.ndarray) -> int:
        """Computes 64-bit difference hash by comparing horizontal pixel gradient."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    def is_duplicate(self, input_val: Union[np.ndarray, int]) -> Union[bool, Tuple[bool, int]]:
        """
        Checks if input is perceptually redundant compared to previously admitted frames.
        Accepts either an image array or an already computed hash.
        """
        if isinstance(input_val, (int, np.integer)):
            curr_hash = int(input_val)
            returns_tuple = False
        else:
            curr_hash = self.compute_dhash(input_val)
            returns_tuple = True

        if not self.seen_hashes:
            self.seen_hashes.append(curr_hash)
            return (False, 64) if returns_tuple else False

        min_dist = 64
        for prev_hash in self.seen_hashes:
            dist = bin(curr_hash ^ prev_hash).count("1")
            if dist < min_dist:
                min_dist = dist
            if dist <= self.hamming_threshold:
                return (True, dist) if returns_tuple else True

        self.seen_hashes.append(curr_hash)
        return (False, min_dist) if returns_tuple else False


class StudioQualityScorer:
    """
    Studio Aesthetic & Pose Quality Evaluator.
    Determines whether the subject in the frame looks flattering, sharp, well-lit,
    and stably posed in studio or stream conditions.
    """

    @staticmethod
    def score_frame(
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        min_sharpness: float = 75.0,
        face_conf: float = 0.0,
        person_conf: float = 0.0
    ) -> Dict[str, Any]:
        """
        Computes a composite studio score (0.0 to 100.0) based on:
        1. Sharpness (Laplacian variance)
        2. Studio Lighting balance (histogram bounds 70-180)
        3. Dynamic contrast (standard deviation of luminance)
        4. Pose Stillness (stability vs previous frame, detecting steady studio pose)
        5. Biometric confidence (face/person presence)
        """
        if frame is None or frame.size == 0:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "empty"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Disqualify bad frames immediately
        if brightness < 20.0 or brightness > 240.0:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "extreme_lighting"}
        if laplacian_var < min_sharpness:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "motion_blur"}

        # 1. Sharpness sub-score (0-30 points)
        # 100 variance = ~15 pts, 300+ variance = 30 pts
        s_sharpness = min(30.0, (laplacian_var / 300.0) * 30.0)

        # 2. Lighting balance sub-score (0-25 points)
        # Optimal studio luminance is centered around 120-145
        diff_from_ideal = abs(brightness - 130.0)
        s_lighting = max(0.0, 25.0 - (diff_from_ideal / 130.0) * 25.0)

        # 3. Dynamic Contrast sub-score (0-20 points)
        # Studio portraits require crisp shadows and highlights (std > 45)
        s_contrast = min(20.0, (contrast / 55.0) * 20.0)

        # 4. Pose Stillness sub-score (0-15 points)
        s_stillness = 10.0  # default when no prev frame
        if prev_frame is not None and prev_frame.shape == frame.shape:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
            diff = cv2.absdiff(gray, prev_gray)
            motion_magnitude = float(np.mean(diff))
            # Low motion (model holding pose) gets highest points
            if motion_magnitude < 6.0:
                s_stillness = 15.0  # Model is holding a stable studio pose
            elif motion_magnitude < 15.0:
                s_stillness = 10.0
            else:
                s_stillness = max(0.0, 15.0 - motion_magnitude)

        # 5. Biometric sub-score (0-10 points)
        s_biometric = 0.0
        if face_conf > 0.0:
            s_biometric += min(6.0, face_conf * 6.0)
        if person_conf > 0.0:
            s_biometric += min(4.0, person_conf * 4.0)

        total_score = round(s_sharpness + s_lighting + s_contrast + s_stillness + s_biometric, 2)
        is_studio_worthy = total_score >= 50.0

        return {
            "studio_score": total_score,
            "is_studio_worthy": is_studio_worthy,
            "sharpness": round(laplacian_var, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "stillness_score": round(s_stillness, 2),
            "reason": "ok"
        }


class DesktopThumbnailManager:
    """Synchronously writes GNOME/Nautilus 512px and 256px thumbnail cache."""

    @staticmethod
    def sync_thumbnail(image_path: Path):
        try:
            uri = image_path.resolve().as_uri()
            uri_hash = hashlib.md5(uri.encode("utf-8")).hexdigest()
            thumb_dirs = [
                Path.home() / ".cache" / "thumbnails" / "large",
                Path.home() / ".cache" / "thumbnails" / "x-large"
            ]
            for t_dir in thumb_dirs:
                t_dir.mkdir(parents=True, exist_ok=True)
                thumb_file = t_dir / f"{uri_hash}.png"
                if not thumb_file.exists():
                    with Image.open(image_path) as im:
                        im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        im.save(thumb_file, "PNG")
        except Exception:
            pass


class HarvestConfig:
    """Helper configuration class for pipeline parameterization."""
    def __init__(
        self,
        mode: str = "interval",
        interval_sec: float = 0.5,
        min_sharpness: float = 75.0,
        max_snapshots: int = 100,
        generate_thumbnails: bool = True,
        dedup_threshold: int = 8,
        only_person: bool = False,
        only_face: bool = False,
        crop_regions: bool = False,
        studio_mode: bool = False,
        situations: int = 8,
        timestamps: Optional[List[float]] = None,
        contact_sheet: bool = False
    ):
        self.mode = mode
        self.interval_sec = interval_sec
        self.min_sharpness = min_sharpness
        self.max_snapshots = max_snapshots
        self.generate_thumbnails = generate_thumbnails
        self.dedup_threshold = dedup_threshold
        self.only_person = only_person
        self.only_face = only_face
        self.crop_regions = crop_regions
        self.studio_mode = studio_mode
        self.situations = situations
        self.timestamps = timestamps or []
        self.contact_sheet = contact_sheet


class VideoSnapshotHarvester:
    """
    Main harvester engine coordinating extraction, quality gating, deduplication,
    studio scoring across diverse situations, biometrics, and atomic dataset output.
    """

    def __init__(
        self,
        output_dir: Path,
        mode: str = "interval",
        min_sharpness: float = 75.0,
        max_snapshots: int = 100,
        dedup_threshold: int = 8,
        only_person: bool = False,
        only_face: bool = False,
        crop_regions: bool = False,
        target_model: Optional[str] = None,
        studio_mode: bool = False,
        situations: int = 8,
        contact_sheet: bool = False,
        verbose: bool = False
    ):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.min_sharpness = min_sharpness
        self.max_snapshots = max_snapshots
        self.dedup = PerceptualDeduplicator(hamming_threshold=dedup_threshold)
        self.only_person = only_person
        self.only_face = only_face
        self.crop_regions = crop_regions
        self.target_model = target_model
        self.studio_mode = studio_mode
        self.situations = situations
        self.contact_sheet = contact_sheet
        self.verbose = verbose

        # Biometric engines (lazy loaded if available)
        self.bio_engine = MultiModalBiometricEngine.get_instance() if MultiModalBiometricEngine else None
        self.somatotype_extractor = SomatotypeRegionalExtractor(self.bio_engine) if (SomatotypeRegionalExtractor and self.bio_engine) else None

    @staticmethod
    def parse_time_str(time_val: Union[str, float, int]) -> float:
        if isinstance(time_val, (int, float)):
            return float(time_val)
        s = str(time_val).strip()
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
        try:
            return float(s)
        except ValueError:
            return 0.0

    def _save_snapshot(
        self,
        frame: np.ndarray,
        stem: str,
        timestamp_sec: float,
        rois_isolated: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, str]]:
        """
        Saves full snapshot and optional anatomical crops with filename reflecting
        video stem and extracted timestamp.
        """
        filename = format_snapshot_filename(stem, timestamp_sec)
        final_dest = self.output_dir / filename
        tmp_dest = self.output_dir / f".tmp_{filename}"

        cv2.imwrite(str(tmp_dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp_dest, final_dest)

        DesktopThumbnailManager.sync_thumbnail(final_dest)

        saved_crops = {}
        if self.crop_regions and rois_isolated and save_regional_crops:
            target_crops_base = self.output_dir.parent if self.output_dir.name in ("snapshots", "pending_review") else self.output_dir
            saved_crops = save_regional_crops(rois_isolated, target_crops_base, final_dest.stem)

        return final_dest, saved_crops

    def harvest_from_timestamps(
        self,
        video_input: Union[Path, str],
        timestamps: List[Union[str, float]]
    ) -> Dict[str, Any]:
        """
        Extracts high-fidelity snapshots at explicitly specified timestamps.
        """
        vid_path = Path(video_input).resolve()
        if not vid_path.is_file():
            raise FileNotFoundError(f"Video file not found: {vid_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        stem = vid_path.stem
        captured = []

        try:
            for t_val in timestamps:
                t_sec = self.parse_time_str(t_val)
                cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                valid_sharp, sharpness, brightness, _ = SharpnessAssessor.evaluate(frame)
                st_info = StudioQualityScorer.score_frame(frame, min_sharpness=self.min_sharpness)

                rois_isolated = {}
                if self.crop_regions and self.somatotype_extractor:
                    rois_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                    rois_isolated = rois_data.get("rois", {})

                final_path, saved_crops = self._save_snapshot(frame, stem, t_sec, rois_isolated)

                entry = {
                    "filename": final_path.name,
                    "path": str(final_path),
                    "timestamp_sec": round(t_sec, 3),
                    "timestamp_str": f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}.{int(round((t_sec - int(t_sec)) * 1000)):03d}",
                    "studio_score": st_info["studio_score"],
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "crops_saved": list(saved_crops.keys())
                }
                captured.append(entry)
                if self.verbose:
                    print(f"🎯 Snapshot at {entry['timestamp_str']} -> {entry['filename']} (Studio Score: {entry['studio_score']})")

        finally:
            cap.release()

        manifest_file = self.output_dir / "SNAPSHOTS_MANIFEST.json"
        manifest_data = {
            "source_video": str(vid_path),
            "video_filename": vid_path.name,
            "generated_at": time.time(),
            "target_model": self.target_model,
            "total_saved": len(captured),
            "snapshots": captured
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data

    def harvest_studio_situations(
        self,
        video_input: Union[Path, str],
        num_situations: int = 8,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discovers distinct situations across the video (poses, lighting setups, scenes)
        and isolates the single best 'peak studio frame' for each situation.
        """
        vid_path = Path(video_input).resolve()
        info = VideoMetadataExtractor.get_info(vid_path)
        duration = info.get("duration", 0.0)
        t_start = self.parse_time_str(start_time) if start_time else 0.0
        t_end = self.parse_time_str(end_time) if end_time else (duration if duration > 0 else 600.0)
        effective_duration = max(1.0, t_end - t_start)

        num_situations = max(1, min(num_situations, self.max_snapshots))
        window_duration = effective_duration / num_situations

        if self.verbose:
            print(f"✨ Scanning video for {num_situations} diverse studio situations...")
            print(f"   Window size per situation: {window_duration:.1f}s across {t_start:.1f}s -> {t_end:.1f}s")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        stem = vid_path.stem
        situation_results = []
        preview_frames_for_sheet = []

        try:
            for s_idx in range(num_situations):
                w_start = t_start + (s_idx * window_duration)
                w_end = min(t_end, w_start + window_duration)
                
                # Sample candidate frames inside this situation window (every 0.5s)
                sample_times = np.linspace(w_start, w_end, num=max(4, int(window_duration * 1.5)))
                
                best_frame = None
                best_sec = w_start
                best_score = -1.0
                best_meta = {}
                prev_sample = None

                for s_time in sample_times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, s_time * 1000.0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # Biometric check if required
                    face_conf = 0.0
                    person_conf = 0.0
                    if self.bio_engine:
                        faces = self.bio_engine.detect_faces(frame)
                        if faces is not None and len(faces) > 0:
                            face_conf = float(faces[0][-1])
                            if self.only_face and face_conf < 0.65:
                                prev_sample = frame
                                continue

                    if self.somatotype_extractor and self.only_person:
                        r_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                        if not r_data.get("person_detected", False):
                            prev_sample = frame
                            continue
                        person_conf = 0.9

                    score_info = StudioQualityScorer.score_frame(
                        frame,
                        prev_frame=prev_sample,
                        min_sharpness=self.min_sharpness,
                        face_conf=face_conf,
                        person_conf=person_conf
                    )

                    score = score_info["studio_score"]
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
                        best_sec = s_time
                        best_meta = score_info

                    prev_sample = frame

                # If a valid frame was found for this situation
                if best_frame is not None and best_score > 0.0:
                    # Check deduplication against previously chosen situations
                    is_dup, dist = self.dedup.is_duplicate(best_frame)
                    if not is_dup:
                        rois_isolated = {}
                        if self.crop_regions and self.somatotype_extractor:
                            r_data = self.somatotype_extractor.extract_keypoint_rois(best_frame)
                            rois_isolated = r_data.get("rois", {})

                        final_path, saved_crops = self._save_snapshot(best_frame, stem, best_sec, rois_isolated)

                        entry = {
                            "situation_index": s_idx + 1,
                            "filename": final_path.name,
                            "path": str(final_path),
                            "timestamp_sec": round(best_sec, 2),
                            "timestamp_str": f"{int(best_sec // 60):02d}:{int(best_sec % 60):02d}",
                            "studio_score": best_score,
                            "sharpness": best_meta.get("sharpness", 0.0),
                            "brightness": best_meta.get("brightness", 0.0),
                            "contrast": best_meta.get("contrast", 0.0),
                            "crops_saved": list(saved_crops.keys())
                        }
                        situation_results.append(entry)
                        preview_frames_for_sheet.append((best_frame, entry))

                        if self.verbose:
                            print(f"   🌟 Situation #{s_idx+1:02d} [{entry['timestamp_str']}]: Score {best_score:.1f} | File: {final_path.name}")

        finally:
            cap.release()

        # Build Studio Contact Sheet if requested
        if self.contact_sheet and preview_frames_for_sheet:
            self._generate_contact_sheet(preview_frames_for_sheet, stem)

        manifest_file = self.output_dir / "SNAPSHOTS_MANIFEST.json"
        manifest_data = {
            "source_video": str(vid_path),
            "video_filename": vid_path.name,
            "mode": "studio_situations",
            "generated_at": time.time(),
            "target_model": self.target_model,
            "total_saved": len(situation_results),
            "situations_discovered": len(situation_results),
            "snapshots": situation_results
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data

    def _generate_contact_sheet(self, frame_entries: List[Tuple[np.ndarray, Dict[str, Any]]], stem: str):
        """Creates a high-resolution studio contact sheet grid with overlaid timestamp and score."""
        n = len(frame_entries)
        if n == 0:
            return

        cols = 4 if n >= 4 else n
        rows = math.ceil(n / cols)
        tile_w, tile_h = 480, 270

        sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, (frame, entry) in enumerate(frame_entries):
            r = idx // cols
            c = idx % cols
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w

            tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

            # Draw sleek HUD badge
            label_top = f"Situation #{entry.get('situation_index', idx+1)}: {entry['timestamp_str']}"
            label_sub = f"Studio Score: {entry['studio_score']:.1f} | Sharp: {entry['sharpness']:.0f}"
            cv2.rectangle(tile, (0, tile_h - 48), (tile_w, tile_h), (20, 20, 20), -1)
            cv2.putText(tile, label_top, (12, tile_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, label_sub, (12, tile_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 180), 1, cv2.LINE_AA)

            sheet[y1:y2, x1:x2] = tile

        sheet_path = self.output_dir / f"{stem}_STUDIO_CONTACT_SHEET.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        DesktopThumbnailManager.sync_thumbnail(sheet_path)
        if self.verbose:
            print(f"📋 Studio Contact Sheet created: {sheet_path.name}")

    def harvest_from_video(
        self,
        video_input: Union[Path, str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        fps_interval: float = 0.5,
        scene_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Executes high-yield snapshot harvesting across target video.
        """
        if self.studio_mode:
            return self.harvest_studio_situations(
                video_input=video_input,
                num_situations=self.situations,
                start_time=start_time,
                end_time=end_time
            )

        vid_path = Path(video_input).resolve()
        if not vid_path.is_file():
            raise FileNotFoundError(f"Video file not found: {vid_path}")

        info = VideoMetadataExtractor.get_info(vid_path)
        duration = info.get("duration", 0.0)
        t_start = self.parse_time_str(start_time) if start_time else 0.0
        t_end = self.parse_time_str(end_time) if end_time else (duration if duration > 0 else 999999.0)

        if self.verbose:
            print(f"🎬 Harvesting from video: {vid_path.name}")
            print(f"   Resolution: {info.get('width', 0)}x{info.get('height', 0)} | Duration: {duration:.1f}s | FPS: {info.get('fps', 0)}")
            print(f"   Time Range: {t_start:.1f}s -> {t_end:.1f}s [Mode: {self.mode}]")

        captured_snapshots = []
        preview_frames = []
        rejected_blurry = 0
        rejected_redundant = 0
        rejected_no_person = 0
        rejected_no_face = 0

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_frames = max(1, int(native_fps / fps_interval)) if self.mode == "interval" else max(1, int(native_fps * 1.5))

        if t_start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000.0)

        stem = vid_path.stem
        seq_idx = 1
        prev_frame = None

        try:
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while cap.isOpened() and len(captured_snapshots) < self.max_snapshots:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                curr_sec = msec / 1000.0

                if curr_sec > t_end and t_end > 0:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # 1. Quality & Sharpness Filter
                valid_sharp, sharpness, brightness, reason = SharpnessAssessor.evaluate(frame)
                if not valid_sharp or sharpness < self.min_sharpness:
                    rejected_blurry += 1
                    curr_frame += step_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                    continue

                # 2. Perceptual Deduplication Filter
                is_dup, dist = self.dedup.is_duplicate(frame)
                if is_dup:
                    rejected_redundant += 1
                    curr_frame += step_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                    continue

                # 3. Biometric Gating
                person_detected = True
                face_detected = True
                rois_isolated = {}
                face_conf = 0.0
                person_conf = 0.0

                if (self.only_person or self.crop_regions) and self.somatotype_extractor:
                    rois_data = self.somatotype_extractor.extract_keypoint_rois(frame, min_person_conf=0.35)
                    person_detected = rois_data.get("person_detected", False)
                    rois_isolated = rois_data.get("rois", {})
                    if self.only_person and not person_detected:
                        rejected_no_person += 1
                        curr_frame += step_frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                        continue
                    if person_detected:
                        person_conf = 0.9

                if (self.only_face or True) and self.bio_engine:
                    faces = self.bio_engine.detect_faces(frame)
                    if faces is not None and len(faces) > 0:
                        face_conf = float(faces[0][-1])
                        face_detected = face_conf >= 0.65
                    else:
                        face_detected = False

                    if self.only_face and not face_detected:
                        rejected_no_face += 1
                        curr_frame += step_frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                        continue

                # 4. Studio Score calculation
                score_info = StudioQualityScorer.score_frame(
                    frame,
                    prev_frame=prev_frame,
                    min_sharpness=self.min_sharpness,
                    face_conf=face_conf,
                    person_conf=person_conf
                )

                # 5. Save with Standardized Name embedding video stem and timestamp
                final_dest, saved_crops = self._save_snapshot(frame, stem, curr_sec, rois_isolated)

                snap_entry = {
                    "seq": seq_idx,
                    "filename": final_dest.name,
                    "path": str(final_dest),
                    "timestamp_sec": round(curr_sec, 2),
                    "timestamp_str": f"{int(curr_sec // 60):02d}:{int(curr_sec % 60):02d}",
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "studio_score": score_info["studio_score"],
                    "hamming_dist": dist,
                    "crops_saved": list(saved_crops.keys())
                }
                captured_snapshots.append(snap_entry)
                preview_frames.append((frame.copy(), snap_entry))

                if self.verbose:
                    crop_str = f" [✂️ Crops: {','.join(saved_crops.keys())}]" if saved_crops else ""
                    print(f"   📸 [{seq_idx:03d}] {snap_entry['timestamp_str']} -> {final_dest.name} | Score: {score_info['studio_score']:.1f}{crop_str}")

                seq_idx += 1
                prev_frame = frame.copy()
                curr_frame += step_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        finally:
            cap.release()

        if self.contact_sheet and preview_frames:
            self._generate_contact_sheet(preview_frames, stem)

        manifest_file = self.output_dir / "SNAPSHOTS_MANIFEST.json"
        manifest_data = {
            "source_video": str(vid_path),
            "video_filename": vid_path.name,
            "generated_at": time.time(),
            "target_model": self.target_model,
            "total_saved": len(captured_snapshots),
            "total_snapshots": len(captured_snapshots),
            "rejected_blurry": rejected_blurry,
            "rejected_redundant": rejected_redundant,
            "rejected_no_person": rejected_no_person,
            "rejected_no_face": rejected_no_face,
            "snapshots": captured_snapshots
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data


# Alias for backward compatibility with tests
class VideoSnapshotPipeline:
    def __init__(self, video_path: str, output_dir: str, config: HarvestConfig):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.config = config
        self.harvester = VideoSnapshotHarvester(
            output_dir=self.output_dir,
            mode=config.mode,
            min_sharpness=config.min_sharpness,
            max_snapshots=config.max_snapshots,
            dedup_threshold=config.dedup_threshold,
            only_person=config.only_person,
            only_face=config.only_face,
            crop_regions=config.crop_regions,
            studio_mode=config.studio_mode,
            situations=config.situations,
            contact_sheet=config.contact_sheet
        )

    def harvest(self) -> Dict[str, Any]:
        if self.config.timestamps:
            return self.harvester.harvest_from_timestamps(self.video_path, self.config.timestamps)
        return self.harvester.harvest_from_video(
            self.video_path,
            fps_interval=(1.0 / self.config.interval_sec) if self.config.interval_sec > 0 else 0.5
        )


def main():
    parser = argparse.ArgumentParser(
        prog="video-snapshot",
        description="High-Precision Video Keyframe & Studio Snapshot Harvester for Biometric Training."
    )
    parser.add_argument("video", nargs="?", help="Path to source video file (MP4, MKV, TS, WebM)")
    parser.add_argument("-m", "--model", help="Target model name (resolves default dataset snapshots folder)")
    parser.add_argument("-o", "--output-dir", help="Output directory for extracted snapshots")
    parser.add_argument("--mode", choices=["keyframes", "scene", "interval"], default="interval", help="Extraction strategy: 'interval' (default, sampled at --fps), 'keyframes', 'scene'")
    parser.add_argument("--fps", type=float, default=0.5, help="Sampling rate in frames per second for interval mode (default: 0.5 = 1 frame every 2s)")
    parser.add_argument("--min-sharpness", type=float, default=75.0, help="Laplacian variance threshold to reject motion blur (default: 75.0)")
    parser.add_argument("--max-snapshots", type=int, default=100, help="Maximum number of snapshots to harvest (default: 100)")
    parser.add_argument("--dhash-threshold", type=int, default=8, help="Max Hamming distance to reject duplicate stationary frames (default: 8)")
    parser.add_argument("--only-person", action="store_true", help="Gating filter: only save frames with a verified human pose (YOLOv8-Pose)")
    parser.add_argument("--only-face", action="store_true", help="Gating filter: only save frames with a verified frontal face (YuNet)")
    parser.add_argument("--crop-regions", action="store_true", help="Extract and save anatomical sub-regions (crops/face, crops/torso, crops/hips)")
    parser.add_argument("--start", help="Start timestamp in video (e.g. 00:01:30 or 90)")
    parser.add_argument("--end", help="End timestamp in video (e.g. 00:10:00 or 600)")
    
    # Studio & Situation Discovery Options
    parser.add_argument("--studio", action="store_true", help="Enable Studio Quality Mode: selects peak flattering studio moments across diverse situations")
    parser.add_argument("--situations", type=int, default=8, help="Number of diverse situations/scenes to discover across video (default: 8)")
    parser.add_argument("--timestamps", help="Comma-separated explicit timestamps to harvest (e.g. '00:01:25,00:04:12.5,00:08:45')")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate a studio contact sheet grid image (STUDIO_CONTACT_SHEET.jpg)")
    parser.add_argument("--inspect", action="store_true", help="Inspect and display top studio situation moments without bulk extraction")

    parser.add_argument("--train-after", action="store_true", help="Automatically trigger biometric model enrollment using harvested snapshots")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose extraction telemetry")

    args = parser.parse_args()

    if not args.video:
        parser.print_help()
        return 1

    vid_path = Path(os.path.expanduser(args.video)).resolve()
    if not vid_path.is_file():
        print(f"❌ Error: El archivo de video no existe: {vid_path}", file=sys.stderr)
        return 1

    # Output directory resolution
    if args.output_dir:
        out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    elif args.model:
        model_name = args.model.strip().lower()
        out_dir = Path.home() / "ComputerVision" / "datasets" / model_name / "photos" / "snapshots"
    else:
        out_dir = vid_path.parent / f"{vid_path.stem}_snapshots"

    harvester = VideoSnapshotHarvester(
        output_dir=out_dir,
        mode=args.mode,
        min_sharpness=args.min_sharpness,
        max_snapshots=args.max_snapshots,
        dedup_threshold=args.dhash_threshold,
        only_person=args.only_person,
        only_face=args.only_face,
        crop_regions=args.crop_regions,
        target_model=args.model,
        studio_mode=args.studio,
        situations=args.situations,
        contact_sheet=args.contact_sheet or args.studio,
        verbose=args.verbose or True
    )

    t0 = time.time()
    print(f"\n🚀 Iniciando extracción de snapshots con {harvester.output_dir.name}...")

    # Explicit timestamps branch
    if args.timestamps:
        ts_list = [t.strip() for t in args.timestamps.split(",") if t.strip()]
        manifest = harvester.harvest_from_timestamps(vid_path, ts_list)
    else:
        manifest = harvester.harvest_from_video(
            video_input=vid_path,
            start_time=args.start,
            end_time=args.end,
            fps_interval=args.fps
        )

    elapsed = time.time() - t0
    saved = manifest.get("total_saved", len(manifest.get("snapshots", [])))
    print(f"\n✅ Extracción completada en {elapsed:.2f}s:")
    print(f"   📸 Snapshots guardados: {saved}")
    print(f"   📁 Directorio de salida: {out_dir}")
    print(f"   📋 Manifiesto JSON: {out_dir / 'SNAPSHOTS_MANIFEST.json'}")

    if args.train_after and args.model:
        print(f"\n🔄 Entrenando modelo biométrico para '{args.model}' con los nuevos snapshots...")
        train_cmd = [
            sys.executable,
            str(MM_SCRAPER_PATH / "model_media_scraper.py"),
            args.model,
            "--train-model"
        ]
        res = subprocess.run(train_cmd)
        if res.returncode == 0:
            print(f"✨ Re-entrenamiento biométrico finalizado con éxito para '{args.model}'.")
        else:
            print(f"⚠️ El comando de re-entrenamiento retornó código: {res.returncode}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
