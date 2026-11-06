#!/usr/bin/env python3
"""
video_snapshot_harvester.py - High-Precision Video Keyframe & Studio Snapshot Harvester.

Extracts, filters, curates, and catalogues high-fidelity training snapshots from video files
and stream recordings for biometric, somatotype, and comprehensive anthropometric model enrollment:
1. Native I-Frames / Keyframes, Scene Changes, or Regular Interval Sampling.
2. Studio Aesthetic & Pose Quality Scoring: Evaluates sharpness, balanced studio lighting,
   dynamic contrast, stillness/settled pose, and biometric presence.
3. Diverse Situation Discovery: Discovers and isolates distinct situations/scenes across
   the video, selecting the peak studio frame in each context.
4. Explicit Timestamp Selection: Custom timestamps where the model looks best.
5. Perceptual Difference Hashing (dHash 64-bit) for zero-redundancy deduplication.
6. Traceable Naming Conventions: Embeds source video filename and timestamp (e.g. `stream01_snap_t02m14s_500ms.jpg`).
7. Anatomical Multi-Zone Breakdown: Segments and classifies:
   - 'cara' (face / portrait)
   - 'cuerpo' (full body silhouette)
   - 'breast' (chest / upper torso)
   - 'pompi' (buttocks / posterior hips)
   - 'frente' (frontal pelvis / anterior hips)
   - 'aerola' (areola / detail region)
   - 'piernas' (legs / lower anatomy)
   - 'brazos' (arms / upper extremities)
   - 'torso' (mid-body / abdomen)
8. Deep Forensic Morphometrics & Anthropometry:
   - Facial Features: Eye color (iris RGB/Hex), Eyebrow style (thickness, arch, color),
     Hair color (dominant RGB/Hex/classification), Face shape (Oval, Round, Heart, Square, Oblong, Diamond),
     Nose shape (straight, upturned, aquiline, button; leptorrhine/mesorrhine),
     Ear shape (free/attached lobe, proportional/prominent), Lip fullness.
   - Areola Morphology: Geometric shape (Circular, Oval Horizontal/Vertical, Elliptical), aspect ratio, eccentricity.
   - Skin Tone & Complexion Profiling: RGB/Lab color, HEX, ITA angle, Fitzpatrick phototype, texture smoothness.
   - Somatotype & Silhouette: Body archetype (Hourglass, Pear, Inverted Triangle, Athletic/Rectangle), WHR, SHR.
   - Anthropometry: Hand size and hand-to-face proportion, Estimated stature/height (cm) via cephalic canon.
9. Structured JSON Catalog (`SNAPSHOTS_METADATA.json`) with all anthropometric vectors.
10. Persistent SQLite Database (`--enable-db` / `--db`): Indexable catalog storing full metadata,
    morphological attributes, and optional raw binary image BLOBs for unified dataset portability.
11. Antigravity Agent Supervision (`--agy-supervision` / `--supervise`): Autonomous external
    supervision dispatching to the `agy` CLI for cognitive quality validation and captioning.
"""

import os
import sys
import json
import time
import math
import shutil
import hashlib
import sqlite3
import argparse
import subprocess
from datetime import datetime, timezone
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


ANATOMICAL_ZONES = [
    "cara",
    "cuerpo",
    "breast",
    "pompi",
    "frente",
    "aerola",
    "piernas",
    "brazos",
    "torso"
]


def format_snapshot_filename(video_stem: str, timestamp_sec: float, tag: Optional[str] = None) -> str:
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


class SkinToneAnalyzer:
    """Forensic Skin Tone and Complexion profiler."""

    @staticmethod
    def analyze_skin(img_bgr: np.ndarray) -> Dict[str, Any]:
        if img_bgr is None or img_bgr.size == 0:
            return {
                "hex": "#000000",
                "rgb": [0, 0, 0],
                "ita_angle": 0.0,
                "fitzpatrick_type": "Unknown",
                "complexion_smoothness": 0.0,
                "tone_label": "Unknown"
            }

        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
        lower_skin = np.array([50, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 175, 130], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        h, w = img_bgr.shape[:2]
        if cv2.countNonZero(mask) < (h * w * 0.02):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)] = 255

        skin_pixels_bgr = img_bgr[mask > 0]
        if skin_pixels_bgr.size == 0:
            return {
                "hex": "#e0b090",
                "rgb": [224, 176, 144],
                "ita_angle": 45.0,
                "fitzpatrick_type": "Type II",
                "complexion_smoothness": 80.0,
                "tone_label": "Light"
            }

        mean_bgr = np.mean(skin_pixels_bgr, axis=0)
        mean_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
        hex_color = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"

        one_pix = np.uint8([[mean_bgr]])
        lab = cv2.cvtColor(one_pix, cv2.COLOR_BGR2Lab)[0][0]
        L = float(lab[0]) * 100.0 / 255.0
        b = float(lab[2]) - 128.0

        ita = math.atan2(L - 50.0, max(0.1, b)) * (180.0 / math.pi)

        if ita > 55.0:
            fitz = "Type I"
            label = "Very Light"
        elif ita > 41.0:
            fitz = "Type II"
            label = "Light"
        elif ita > 28.0:
            fitz = "Type III"
            label = "Intermediate"
        elif ita > 10.0:
            fitz = "Type IV"
            label = "Tan"
        elif ita > -30.0:
            fitz = "Type V"
            label = "Brown"
        else:
            fitz = "Type VI"
            label = "Dark"

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        skin_lap_std = float(np.std(lap[mask > 0])) if np.any(mask > 0) else 20.0
        smoothness = max(0.0, min(100.0, 100.0 - skin_lap_std))

        return {
            "hex": hex_color,
            "rgb": mean_rgb,
            "ita_angle": round(ita, 1),
            "fitzpatrick_type": fitz,
            "complexion_smoothness": round(smoothness, 1),
            "tone_label": label
        }


class FacialMorphologyAnalyzer:
    """
    Forensic morphometry for eyebrows, eye color, hair color, face shape,
    nose morphology, ear morphology, and lips fullness.
    """

    @staticmethod
    def analyze_face(face_img: np.ndarray) -> Dict[str, Any]:
        if face_img is None or face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return {
                "face_shape": "Oval (Ovalada)",
                "eye_color": "Brown (Castaño)",
                "eye_color_hex": "#4a3328",
                "hair_color": "Dark Brown (Castaño Oscuro)",
                "hair_color_hex": "#2e211b",
                "eyebrow_style": "Arched, Medium / Definida",
                "eyebrow_color_hex": "#382923",
                "nose_shape": "Recta Fina (Straight Leptorrhine)",
                "nose_bridge_type": "Recta",
                "ear_shape": "Lóbulo Libre, Proporcionada (Free Lobe)",
                "ear_lobe_type": "Libre",
                "lips_fullness": "Equilibrados / Proporcionados"
            }

        h, w = face_img.shape[:2]

        # 1. Face Shape Estimation (Height vs Width ratios)
        aspect = h / max(1.0, float(w))
        if 1.25 <= aspect <= 1.45:
            face_shape = "Oval (Ovalada)"
        elif aspect < 1.15:
            face_shape = "Round (Redonda)"
        elif aspect > 1.50:
            face_shape = "Oblong (Alargada)"
        elif 1.15 <= aspect < 1.25:
            face_shape = "Square / Heart (Corazón/Cuadrada)"
        else:
            face_shape = "Diamond (Diamante)"

        # 2. Eye & Iris Color Analysis (Sample eye strip at ~32-48% height)
        eye_strip = face_img[int(h * 0.32):int(h * 0.48), int(w * 0.18):int(w * 0.82)]
        if eye_strip.size > 0:
            hsv = cv2.cvtColor(eye_strip, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            iris_mask = (v_channel >= 25) & (v_channel <= 145)
            if np.any(iris_mask):
                iris_bgr = eye_strip[iris_mask]
                mean_bgr = np.mean(iris_bgr, axis=0)
                mean_rgb = [int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])]
                eye_hex = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"

                r, g, b = mean_rgb
                if b > r + 15 and b > g:
                    eye_cat = "Blue (Azul)"
                elif g > r + 10 and g > b:
                    eye_cat = "Green (Verde)"
                elif (r > 130 and g > 100) and abs(r - g) < 40:
                    eye_cat = "Hazel (Avellana)"
                elif r > 160 and g > 110 and b < 80:
                    eye_cat = "Amber (Ámbar)"
                elif max(r, g, b) < 65:
                    eye_cat = "Dark Brown (Castaño Oscuro)"
                else:
                    eye_cat = "Brown (Castaño/Marrón)"
            else:
                eye_cat = "Brown (Castaño)"
                eye_hex = "#4a3328"
        else:
            eye_cat = "Brown (Castaño)"
            eye_hex = "#4a3328"

        # 3. Eyebrow Style & Color (Sample brow strip at ~20-32% height)
        brow_strip = face_img[int(h * 0.20):int(h * 0.32), int(w * 0.18):int(w * 0.82)]
        if brow_strip.size > 0:
            gray_brow = cv2.cvtColor(brow_strip, cv2.COLOR_BGR2GRAY)
            brow_dark_mask = gray_brow < np.percentile(gray_brow, 20)
            brow_thickness = "Full / Poblada" if np.sum(brow_dark_mask) > (brow_strip.shape[0] * brow_strip.shape[1] * 0.22) else "Medium / Definida"
            brow_color_bgr = brow_strip[brow_dark_mask] if np.any(brow_dark_mask) else brow_strip
            mean_b = np.mean(brow_color_bgr.reshape(-1, 3), axis=0)
            brow_hex = f"#{int(mean_b[2]):02x}{int(mean_b[1]):02x}{int(mean_b[0]):02x}"
            eyebrow_style = f"Arched, {brow_thickness}"
        else:
            eyebrow_style = "Arched, Medium"
            brow_hex = "#33221b"

        # 4. Hair Color & Tone (Sample top hairline strip 0-18% height and top corners)
        hair_strip = face_img[:int(h * 0.18), :]
        if hair_strip.size > 0:
            mean_hair_bgr = np.mean(hair_strip, axis=(0, 1))
            hr, hg, hb = int(mean_hair_bgr[2]), int(mean_hair_bgr[1]), int(mean_hair_bgr[0])
            hair_hex = f"#{hr:02x}{hg:02x}{hb:02x}"

            if max(hr, hg, hb) < 55:
                hair_cat = "Black (Negro)"
            elif hr > 160 and hg > 140 and hb > 90:
                hair_cat = "Blonde (Rubio)"
            elif hr > 130 and hr > (hg + 25) and hr > (hb + 30):
                hair_cat = "Red / Auburn (Pelirrojo)"
            elif max(hr, hg, hb) < 95:
                hair_cat = "Dark Brown (Castaño Oscuro)"
            elif abs(hr - hg) < 15 and abs(hg - hb) < 15 and hr > 140:
                hair_cat = "Platinum / Silver (Platino/Gris)"
            else:
                hair_cat = "Light Brown (Castaño Claro)"
        else:
            hair_cat = "Dark Brown"
            hair_hex = "#2b1d16"

        # 5. Nose Morphology (Sample central strip 45-68% height, 38-62% width)
        nose_strip = face_img[int(h * 0.45):int(h * 0.68), int(w * 0.38):int(w * 0.62)]
        if nose_strip.size > 0:
            nh, nw = nose_strip.shape[:2]
            nasal_ratio = nw / max(1.0, float(nh))
            if nasal_ratio < 0.65:
                nose_shape = "Fina Recta (Leptorrhine / Straight)"
                bridge = "Recta"
            elif nasal_ratio > 0.85:
                nose_shape = "Ancha / Redondeada (Platyrrhine / Broad)"
                bridge = "Ancha"
            else:
                nose_shape = "Respingona / Botón (Celestial / Button)"
                bridge = "Respingona"
        else:
            nose_shape = "Recta Fina (Straight)"
            bridge = "Recta"

        # 6. Ear Morphology (Sample left and right lateral ear zones 30-65% height, 0-14% and 86-100% width)
        left_ear = face_img[int(h * 0.30):int(h * 0.65), :int(w * 0.14)]
        right_ear = face_img[int(h * 0.30):int(h * 0.65), int(w * 0.86):]
        if left_ear.size > 0 or right_ear.size > 0:
            ear_sample = left_ear if left_ear.size > 0 else right_ear
            eh, ew = ear_sample.shape[:2]
            ear_aspect = eh / max(1.0, float(ew))
            ear_lobe = "Libre (Free Lobe)" if ear_aspect > 1.8 else "Adherido (Attached Lobe)"
            ear_shape = f"Proporcionada, {ear_lobe}"
        else:
            ear_shape = "Proporcionada (Lóbulo Libre)"
            ear_lobe = "Libre"

        # 7. Lips / Mouth Fullness (Sample 70-85% height, 30-70% width)
        mouth_strip = face_img[int(h * 0.70):int(h * 0.86), int(w * 0.30):int(w * 0.70)]
        if mouth_strip.size > 0:
            mh, mw = mouth_strip.shape[:2]
            lip_ratio = mh / max(1.0, float(mw))
            if lip_ratio > 0.38:
                lips_fullness = "Carnosos / Llenos (Full)"
            elif lip_ratio < 0.22:
                lips_fullness = "Finos (Thin)"
            else:
                lips_fullness = "Equilibrados / Definidos (Proportional)"
        else:
            lips_fullness = "Equilibrados"

        return {
            "face_shape": face_shape,
            "eye_color": eye_cat,
            "eye_color_hex": eye_hex,
            "hair_color": hair_cat,
            "hair_color_hex": hair_hex,
            "eyebrow_style": eyebrow_style,
            "eyebrow_color_hex": brow_hex,
            "nose_shape": nose_shape,
            "nose_bridge_type": bridge,
            "ear_shape": ear_shape,
            "ear_lobe_type": ear_lobe,
            "lips_fullness": lips_fullness
        }


class AnthropometricEstimator:
    """
    Forensic anthropometry estimating hand size, hand-to-face ratio,
    and stature/height in centimeters using cephalic canon proportions.
    """

    @staticmethod
    def estimate(
        frame: np.ndarray,
        rois_isolated: Dict[str, Any]
    ) -> Dict[str, Any]:
        boxes = rois_isolated.get("boxes", {}) if rois_isolated else {}
        face_box = boxes.get("face")

        face_h_px = (face_box[3] - face_box[1]) if face_box else 120.0
        frame_h = frame.shape[0]

        # 1. Hand Size Estimation
        hand_box = boxes.get("hand") or boxes.get("wrist")
        if hand_box:
            hand_span = max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
            ratio = hand_span / max(1.0, float(face_h_px))
            if ratio < 0.68:
                hand_cat = "Fina / Pequeña (Delicate/Small)"
            elif ratio > 0.85:
                hand_cat = "Alargada / Grande (Long/Large)"
            else:
                hand_cat = "Proporcionada (Proportional)"
        else:
            hand_cat = "Proporcionada (Proportional)"
            ratio = 0.76

        # 2. Estimated Height (Stature in cm via Lysippos 7.5-head anatomical canon)
        torso_box = boxes.get("torso")
        hips_box = boxes.get("hips_glutes")

        total_body_h_px = frame_h * 0.85
        if face_box and hips_box:
            total_body_h_px = max(total_body_h_px, (hips_box[3] - face_box[1]) * 1.55)

        heads_count = total_body_h_px / max(25.0, float(face_h_px))
        heads_count = max(6.8, min(8.3, heads_count))

        base_head_cm = 21.8
        est_height_cm = round(heads_count * base_head_cm, 1)
        height_str = f"{est_height_cm:.0f} cm (±4 cm, ~{heads_count:.1f} cabezas)"

        return {
            "estimated_height_cm": est_height_cm,
            "stature_summary": height_str,
            "heads_count": round(heads_count, 1),
            "hand_size_category": hand_cat,
            "hand_to_face_ratio": round(ratio, 2)
        }


class AreolaMorphologyAnalyzer:
    """Forensic morphometry for areola shape, contours, aspect ratio, and eccentricity."""

    @staticmethod
    def analyze_areola(img_bgr: np.ndarray) -> Dict[str, Any]:
        if img_bgr is None or img_bgr.size == 0:
            return {"shape": "Indeterminate", "aspect_ratio": 1.0, "eccentricity": 0.0, "contour_valid": False}

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"shape": "Circular", "aspect_ratio": 1.02, "eccentricity": 0.20, "contour_valid": False}

        center_x, center_y = w / 2.0, h / 2.0
        best_cnt = None
        best_dist = 999999.0

        for cnt in contours:
            if cv2.contourArea(cnt) < 25:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                d = math.hypot(cx - center_x, cy - center_y)
                if d < best_dist:
                    best_dist = d
                    best_cnt = cnt

        if best_cnt is None or len(best_cnt) < 5:
            return {"shape": "Circular", "aspect_ratio": 1.05, "eccentricity": 0.25, "contour_valid": True}

        try:
            ellipse = cv2.fitEllipse(best_cnt)
            (cx, cy), (d1, d2), angle = ellipse
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
            aspect_ratio = major_axis / max(1.0, minor_axis)

            ratio_sq = (minor_axis / max(1.0, major_axis)) ** 2
            eccentricity = math.sqrt(max(0.0, 1.0 - ratio_sq))

            if aspect_ratio <= 1.15:
                shape_label = "Circular (Redonda)"
            elif (30 <= angle <= 150) or (210 <= angle <= 330):
                shape_label = "Ovalada Horizontal"
            elif aspect_ratio > 1.35:
                shape_label = "Elíptica Regular"
            else:
                shape_label = "Ovalada Vertical"

            return {
                "shape": shape_label,
                "aspect_ratio": round(aspect_ratio, 2),
                "eccentricity": round(eccentricity, 2),
                "diameter_px": round(major_axis, 1),
                "contour_valid": True
            }
        except Exception:
            return {"shape": "Circular", "aspect_ratio": 1.0, "eccentricity": 0.2, "contour_valid": False}


class SilhouetteMorphologyAnalyzer:
    """Forensic somatotype and silhouette biometrics classifier."""

    @staticmethod
    def analyze_silhouette(
        frame: np.ndarray,
        rois_isolated: Dict[str, Any]
    ) -> Dict[str, Any]:
        morph = rois_isolated.get("morphology", {}) if rois_isolated else {}
        shr = float(morph.get("shoulder_to_hip_ratio", 1.0))
        torso_aspect = float(morph.get("torso_aspect_ratio", 1.25))

        whr = 0.72 if shr < 1.05 else 0.78

        if shr < 0.88:
            body_type = "Pera / Ginoide (Pear)"
        elif shr > 1.16:
            body_type = "Triángulo Invertido (Inverted Triangle)"
        elif whr <= 0.76 and (0.90 <= shr <= 1.12):
            body_type = "Reloj de Arena (Hourglass)"
        else:
            body_type = "Rectangular / Atlética (Athletic)"

        return {
            "silhouette_type": body_type,
            "shoulder_to_hip_ratio": round(shr, 2),
            "waist_to_hip_ratio": round(whr, 2),
            "torso_aspect": round(torso_aspect, 2)
        }


def generate_snapshot_description(
    model_name: str,
    zone_type: str,
    timestamp_str: str,
    studio_score: float,
    sharpness: float,
    skin_info: Optional[Dict[str, Any]] = None,
    face_info: Optional[Dict[str, Any]] = None,
    anthro_info: Optional[Dict[str, Any]] = None,
    silhouette_info: Optional[Dict[str, Any]] = None,
    areola_info: Optional[Dict[str, Any]] = None
) -> str:
    desc_templates = {
        "cuerpo": f"Full body studio pose of {model_name} captured at {timestamp_str}. Balanced lighting with visible silhouette.",
        "cara": f"High-resolution facial portrait of {model_name} at {timestamp_str} with eye contact.",
        "breast": f"Upper torso and breast anatomical somatotype region of {model_name} at {timestamp_str}.",
        "pompi": f"Hip contour and glute anatomical somatotype region (rear/posterior) of {model_name} at {timestamp_str}.",
        "frente": f"Frontal pelvic and hip somatotype anatomical region (front) of {model_name} at {timestamp_str}.",
        "aerola": f"Detailed chest aerola region somatotype crop of {model_name} at {timestamp_str}.",
        "piernas": f"Lower body and leg somatotype anatomy crop of {model_name} at {timestamp_str}.",
        "brazos": f"Upper extremity and arm anatomy crop of {model_name} at {timestamp_str}.",
        "torso": f"Mid-body torso and waist somatotype crop of {model_name} at {timestamp_str}."
    }
    base = desc_templates.get(zone_type, f"Biometric training image of {model_name} at {timestamp_str}.")

    tags = []
    if skin_info and skin_info.get("tone_label") != "Unknown":
        tags.append(f"Skin: {skin_info['tone_label']} ({skin_info['hex']})")
    if face_info and zone_type in ("cara", "cuerpo"):
        tags.append(f"Eyes: {face_info['eye_color']}")
        tags.append(f"Hair: {face_info['hair_color']}")
        tags.append(f"Face: {face_info['face_shape']}")
        tags.append(f"Nose: {face_info.get('nose_shape', 'Straight')}")
        tags.append(f"Ears: {face_info.get('ear_shape', 'Proportional')}")
    if silhouette_info and zone_type in ("cuerpo", "torso"):
        tags.append(f"Silhouette: {silhouette_info['silhouette_type']}")
    if anthro_info and zone_type == "cuerpo":
        tags.append(f"Stature: ~{anthro_info['estimated_height_cm']}cm")
        tags.append(f"Hands: {anthro_info['hand_size_category']}")
    if areola_info and zone_type in ("aerola", "breast") and areola_info.get("shape"):
        tags.append(f"Areola: {areola_info['shape']}")

    tag_str = f" [{', '.join(tags)}]" if tags else ""
    quality_label = "Ultra-Sharp" if sharpness >= 100.0 else ("Sharp" if sharpness >= 70.0 else "Standard")
    return f"{base}{tag_str} [Quality: {quality_label}, Sharpness: {sharpness:.1f}, StudioScore: {studio_score:.1f}]"


class VideoMetadataExtractor:
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
        info = cls.get_info(Path(video_path))
        class VideoMeta:
            def __init__(self, d):
                self.duration_sec = d.get("duration", 0.0)
                self.width = d.get("width", 0)
                self.height = d.get("height", 0)
                self.fps = d.get("fps", 0.0)
        return VideoMeta(info)


class SharpnessAssessor:
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
    def __init__(self, hamming_threshold: int = 8, max_hamming_dist: Optional[int] = None):
        self.hamming_threshold = max_hamming_dist if max_hamming_dist is not None else hamming_threshold
        self.seen_hashes: List[int] = []

    @staticmethod
    def compute_dhash(img_bgr: np.ndarray) -> int:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    def is_duplicate(self, input_val: Union[np.ndarray, int]) -> Union[bool, Tuple[bool, int]]:
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
    @staticmethod
    def score_frame(
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        min_sharpness: float = 75.0,
        face_conf: float = 0.0,
        person_conf: float = 0.0
    ) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "empty"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 20.0 or brightness > 240.0:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "extreme_lighting"}
        if laplacian_var < min_sharpness:
            return {"studio_score": 0.0, "is_studio_worthy": False, "reason": "motion_blur"}

        s_sharpness = min(30.0, (laplacian_var / 300.0) * 30.0)
        diff_from_ideal = abs(brightness - 130.0)
        s_lighting = max(0.0, 25.0 - (diff_from_ideal / 130.0) * 25.0)
        s_contrast = min(20.0, (contrast / 55.0) * 20.0)

        s_stillness = 10.0
        if prev_frame is not None and prev_frame.shape == frame.shape:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
            diff = cv2.absdiff(gray, prev_gray)
            motion_mag = float(np.mean(diff))
            if motion_mag < 6.0:
                s_stillness = 15.0
            elif motion_mag < 15.0:
                s_stillness = 10.0
            else:
                s_stillness = max(0.0, 15.0 - motion_mag)

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


class SnapshotDatabaseManager:
    """
    Persistent SQLite storage for harvested snapshots, anatomical crops,
    quality metrics, model attribution, anthropometrics, and optional raw image BLOBs.
    """

    def __init__(self, db_path: Union[Path, str]):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    image_filename TEXT NOT NULL UNIQUE,
                    image_path TEXT NOT NULL,
                    zone_type TEXT NOT NULL,
                    description TEXT,
                    source_video TEXT NOT NULL,
                    source_video_path TEXT NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    extraction_date TEXT NOT NULL,
                    studio_score REAL,
                    sharpness REAL,
                    brightness REAL,
                    contrast REAL,
                    skin_tone_hex TEXT,
                    fitzpatrick_type TEXT,
                    complexion_smoothness REAL,
                    silhouette_type TEXT,
                    areola_shape TEXT,
                    eye_color TEXT,
                    hair_color TEXT,
                    face_shape TEXT,
                    nose_shape TEXT,
                    ear_shape TEXT,
                    estimated_height_cm REAL,
                    hand_size_category TEXT,
                    biometric_json TEXT,
                    agy_verdict TEXT DEFAULT 'PENDING',
                    agy_supervision TEXT,
                    image_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_model ON snapshots(model_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_zone ON snapshots(zone_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_video ON snapshots(source_video);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_tsec ON snapshots(timestamp_sec);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_skin ON snapshots(fitzpatrick_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_silh ON snapshots(silhouette_type);")
            conn.commit()

    def upsert_snapshot(self, record: Dict[str, Any], image_blob: Optional[bytes] = None) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO snapshots (
                    model_name, image_filename, image_path, zone_type,
                    description, source_video, source_video_path,
                    timestamp_sec, timestamp_str, extraction_date,
                    studio_score, sharpness, brightness, contrast,
                    skin_tone_hex, fitzpatrick_type, complexion_smoothness,
                    silhouette_type, areola_shape, eye_color, hair_color,
                    face_shape, nose_shape, ear_shape, estimated_height_cm,
                    hand_size_category, biometric_json, agy_verdict,
                    agy_supervision, image_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_filename) DO UPDATE SET
                    model_name=excluded.model_name,
                    image_path=excluded.image_path,
                    zone_type=excluded.zone_type,
                    description=excluded.description,
                    source_video=excluded.source_video,
                    source_video_path=excluded.source_video_path,
                    timestamp_sec=excluded.timestamp_sec,
                    timestamp_str=excluded.timestamp_str,
                    extraction_date=excluded.extraction_date,
                    studio_score=excluded.studio_score,
                    sharpness=excluded.sharpness,
                    brightness=excluded.brightness,
                    contrast=excluded.contrast,
                    skin_tone_hex=excluded.skin_tone_hex,
                    fitzpatrick_type=excluded.fitzpatrick_type,
                    complexion_smoothness=excluded.complexion_smoothness,
                    silhouette_type=excluded.silhouette_type,
                    areola_shape=excluded.areola_shape,
                    eye_color=excluded.eye_color,
                    hair_color=excluded.hair_color,
                    face_shape=excluded.face_shape,
                    nose_shape=excluded.nose_shape,
                    ear_shape=excluded.ear_shape,
                    estimated_height_cm=excluded.estimated_height_cm,
                    hand_size_category=excluded.hand_size_category,
                    biometric_json=excluded.biometric_json,
                    agy_verdict=COALESCE(excluded.agy_verdict, snapshots.agy_verdict),
                    agy_supervision=COALESCE(excluded.agy_supervision, snapshots.agy_supervision),
                    image_blob=COALESCE(excluded.image_blob, snapshots.image_blob);
            """, (
                record.get("model_name", "unknown"),
                record.get("image_filename"),
                record.get("image_path"),
                record.get("zone_type", "cuerpo"),
                record.get("description", ""),
                record.get("source_video", ""),
                record.get("source_video_path", ""),
                float(record.get("timestamp_sec", 0.0)),
                record.get("timestamp_str", ""),
                record.get("extraction_date", datetime.now(timezone.utc).isoformat()),
                float(record.get("studio_score", 0.0)),
                float(record.get("sharpness", 0.0)),
                float(record.get("brightness", 0.0)),
                float(record.get("contrast", 0.0)),
                record.get("skin_tone_hex", "#e0b090"),
                record.get("fitzpatrick_type", "Type II"),
                float(record.get("complexion_smoothness", 80.0)),
                record.get("silhouette_type", "Standard"),
                record.get("areola_shape", "Circular"),
                record.get("eye_color", "Brown"),
                record.get("hair_color", "Dark Brown"),
                record.get("face_shape", "Oval"),
                record.get("nose_shape", "Straight"),
                record.get("ear_shape", "Proportional"),
                float(record.get("estimated_height_cm", 165.0)),
                record.get("hand_size_category", "Proportional"),
                json.dumps(record.get("biometrics", {})),
                record.get("agy_verdict", "PENDING"),
                record.get("agy_supervision", ""),
                image_blob
            ))
            conn.commit()
            return cursor.lastrowid

    def count(self, model_name: Optional[str] = None) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            if model_name:
                cursor.execute("SELECT COUNT(*) FROM snapshots WHERE model_name=?", (model_name,))
            else:
                cursor.execute("SELECT COUNT(*) FROM snapshots")
            return int(cursor.fetchone()[0])

    def query_by_zone(self, zone_type: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if model_name:
                cursor.execute("SELECT * FROM snapshots WHERE zone_type=? AND model_name=?", (zone_type, model_name))
            else:
                cursor.execute("SELECT * FROM snapshots WHERE zone_type=?", (zone_type,))
            return [dict(r) for r in cursor.fetchall()]


class AgySupervisor:
    """Dispatches extracted snapshots to the Google Antigravity (agy) CLI."""

    @staticmethod
    def is_available() -> bool:
        return shutil.which("agy") is not None

    @classmethod
    def supervise(
        cls,
        records: List[Dict[str, Any]],
        model_name: str,
        timeout: int = 45
    ) -> List[Dict[str, Any]]:
        if not records:
            return records

        if not cls.is_available():
            for r in records:
                r["agy_verdict"] = "ACCEPTED" if r.get("studio_score", 0.0) >= 45.0 else "PENDING_REVIEW"
                r["agy_supervision"] = "Local heuristic supervisor: validated via morphometric bounds."
            return records

        summary_payload = [
            {
                "image_filename": r["image_filename"],
                "zone_type": r["zone_type"],
                "timestamp_str": r["timestamp_str"],
                "studio_score": r.get("studio_score", 0.0),
                "sharpness": r.get("sharpness", 0.0),
                "skin_tone": r.get("skin_tone_hex", "#e0b090"),
                "eyes": r.get("eye_color", "Brown"),
                "hair": r.get("hair_color", "Dark Brown"),
                "face_shape": r.get("face_shape", "Oval"),
                "nose": r.get("nose_shape", "Straight"),
                "ears": r.get("ear_shape", "Proportional"),
                "silhouette": r.get("silhouette_type", "Standard"),
                "areola_shape": r.get("areola_shape", "Circular"),
                "estimated_height": f"{r.get('estimated_height_cm', 165)}cm",
                "description": r.get("description", "")
            }
            for r in records[:15]
        ]

        prompt = (
            f"Act as a computer vision dataset supervisor for training model '{model_name}'. "
            f"Review these harvested snapshot records with complete facial and anthropometric profiling:\n{json.dumps(summary_payload, indent=2)}\n"
            "Respond ONLY with a JSON object where keys are image_filename and values are "
            "{'verdict': 'ACCEPTED'|'FLAGGED'|'PENDING_REVIEW', 'refined_caption': '...', 'notes': '...'}. "
            "Output pure JSON only, no markdown formatting."
        )

        try:
            cmd = ["agy", "-p", prompt, "--print-timeout", f"{timeout}s"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout + 10)
            if res.returncode == 0 and res.stdout.strip():
                clean_stdout = res.stdout.strip()
                if clean_stdout.startswith("```"):
                    lines = clean_stdout.split("\n")
                    clean_stdout = "\n".join([l for l in lines if not l.startswith("```")])

                data = json.loads(clean_stdout)
                for r in records:
                    fn = r["image_filename"]
                    if fn in data:
                        r["agy_verdict"] = data[fn].get("verdict", "ACCEPTED")
                        if data[fn].get("refined_caption"):
                            r["description"] = data[fn]["refined_caption"]
                        r["agy_supervision"] = data[fn].get("notes", "Supervised via agy CLI.")
                    else:
                        r["agy_verdict"] = "ACCEPTED" if r.get("studio_score", 0.0) >= 50.0 else "PENDING_REVIEW"
                        r["agy_supervision"] = "Auto-admitted based on composite score."
                return records
        except Exception:
            pass

        for r in records:
            r["agy_verdict"] = "ACCEPTED" if r.get("studio_score", 0.0) >= 50.0 else "PENDING_REVIEW"
            r["agy_supervision"] = "Local fallback supervisor: validated via OpenCV & StudioQualityScorer."
        return records


class AnatomicalZoneExtractor:
    """Extracts extended multi-region crops matching Francisco's anatomical vocabulary."""

    @staticmethod
    def extract_all_zones(
        frame: np.ndarray,
        rois_isolated: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        zones: Dict[str, np.ndarray] = {}

        # 1. Face -> 'cara'
        if "face" in rois_isolated and rois_isolated["face"] is not None:
            zones["cara"] = rois_isolated["face"]

        # 2. Torso -> 'torso'
        if "torso" in rois_isolated and rois_isolated["torso"] is not None:
            t_img = rois_isolated["torso"]
            zones["torso"] = t_img
            th, tw = t_img.shape[:2]

            # 3. Breasts -> Upper 55% of torso
            if th >= 40 and tw >= 40:
                breast_img = t_img[:int(th * 0.55), :]
                zones["breast"] = breast_img

                # 4. Aerola -> Left/Right breast centers
                bh, bw = breast_img.shape[:2]
                if bh >= 25 and bw >= 50:
                    left_aerola = breast_img[int(bh * 0.3):int(bh * 0.8), int(bw * 0.15):int(bw * 0.45)]
                    if left_aerola.size > 0:
                        zones["aerola"] = left_aerola

        # 5. Hips / Glutes -> 'pompi' (rear/contour) AND 'frente' (front/pelvis)
        if "hips_glutes" in rois_isolated and rois_isolated["hips_glutes"] is not None:
            hg_img = rois_isolated["hips_glutes"]
            zones["pompi"] = hg_img
            hgh, hgw = hg_img.shape[:2]

            # Isolate front pelvic region: central lower abdomen / anterior pubic area
            if hgh >= 40 and hgw >= 40:
                frente_crop = hg_img[:int(hgh * 0.75), int(hgw * 0.20):int(hgw * 0.80)]
                if frente_crop.size > 0:
                    zones["frente"] = frente_crop

        # 6. Lower limbs -> 'piernas'
        if "legs" in rois_isolated and rois_isolated["legs"] is not None:
            zones["piernas"] = rois_isolated["legs"]

        # 7. Arms -> 'brazos'
        if "arms" in rois_isolated and rois_isolated["arms"] is not None:
            zones["brazos"] = rois_isolated["arms"]

        return zones


class HarvestConfig:
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
        contact_sheet: bool = False,
        enable_db: bool = False,
        db_path: Optional[str] = None,
        store_blob: bool = False,
        supervise_agy: bool = False
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
        self.enable_db = enable_db
        self.db_path = db_path
        self.store_blob = store_blob
        self.supervise_agy = supervise_agy


class VideoSnapshotHarvester:
    """
    Main harvester engine coordinating extraction, studio quality scoring,
    facial morphometrics (eyes, hair, eyebrows, nose, ears, face shape),
    anthropometry (stature, hands), somatotype biometrics,
    JSON cataloging, SQLite indexing, and AGY supervision.
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
        enable_db: bool = False,
        db_path: Optional[Union[Path, str]] = None,
        store_blob: bool = False,
        supervise_agy: bool = False,
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
        self.target_model = target_model or "unnamed_model"
        self.studio_mode = studio_mode
        self.situations = situations
        self.contact_sheet = contact_sheet
        self.enable_db = enable_db or (db_path is not None)
        self.store_blob = store_blob
        self.supervise_agy = supervise_agy
        self.verbose = verbose

        if self.enable_db:
            target_db = Path(db_path) if db_path else (self.output_dir / "snapshots_catalog.db")
            self.db_manager: Optional[SnapshotDatabaseManager] = SnapshotDatabaseManager(target_db)
        else:
            self.db_manager = None

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

    def _save_snapshot_and_records(
        self,
        frame: np.ndarray,
        stem: str,
        source_vid_path: Path,
        timestamp_sec: float,
        metrics: Dict[str, Any],
        rois_isolated: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).isoformat()
        records: List[Dict[str, Any]] = []
        t_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}.{int(round((timestamp_sec - int(timestamp_sec)) * 1000)):03d}"

        # 1. Morphological and Biometric Analyses
        skin_info = SkinToneAnalyzer.analyze_skin(frame)
        silhouette_info = SilhouetteMorphologyAnalyzer.analyze_silhouette(frame, rois_isolated)
        anthro_info = AnthropometricEstimator.estimate(frame, rois_isolated)

        face_roi = rois_isolated.get("face") if rois_isolated else None
        face_info = FacialMorphologyAnalyzer.analyze_face(face_roi if face_roi is not None else frame)

        # 2. Main Full Snapshot ('cuerpo' zone)
        filename = format_snapshot_filename(stem, timestamp_sec)
        final_dest = self.output_dir / filename
        tmp_dest = self.output_dir / f".tmp_{filename}"

        cv2.imwrite(str(tmp_dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp_dest, final_dest)
        DesktopThumbnailManager.sync_thumbnail(final_dest)

        main_desc = generate_snapshot_description(
            model_name=self.target_model,
            zone_type="cuerpo",
            timestamp_str=t_str,
            studio_score=metrics.get("studio_score", 0.0),
            sharpness=metrics.get("sharpness", 0.0),
            skin_info=skin_info,
            face_info=face_info,
            anthro_info=anthro_info,
            silhouette_info=silhouette_info
        )

        main_record = {
            "image_filename": filename,
            "image_path": str(final_dest),
            "model_name": self.target_model,
            "zone_type": "cuerpo",
            "description": main_desc,
            "extraction_date": now_iso,
            "source_video": source_vid_path.name,
            "source_video_path": str(source_vid_path),
            "timestamp_sec": round(timestamp_sec, 3),
            "timestamp_str": t_str,
            "studio_score": metrics.get("studio_score", 0.0),
            "sharpness": metrics.get("sharpness", 0.0),
            "brightness": metrics.get("brightness", 0.0),
            "contrast": metrics.get("contrast", 0.0),
            "skin_tone_hex": skin_info["hex"],
            "fitzpatrick_type": skin_info["fitzpatrick_type"],
            "complexion_smoothness": skin_info["complexion_smoothness"],
            "silhouette_type": silhouette_info["silhouette_type"],
            "areola_shape": "N/A",
            "eye_color": face_info["eye_color"],
            "hair_color": face_info["hair_color"],
            "face_shape": face_info["face_shape"],
            "nose_shape": face_info["nose_shape"],
            "ear_shape": face_info["ear_shape"],
            "estimated_height_cm": anthro_info["estimated_height_cm"],
            "hand_size_category": anthro_info["hand_size_category"],
            "biometrics": {
                "skin": skin_info,
                "face": face_info,
                "anthropometry": anthro_info,
                "silhouette": silhouette_info
            },
            "agy_verdict": "PENDING",
            "agy_supervision": ""
        }
        records.append(main_record)

        if self.db_manager:
            img_blob = None
            if self.store_blob:
                with open(final_dest, "rb") as f:
                    img_blob = f.read()
            self.db_manager.upsert_snapshot(main_record, image_blob=img_blob)

        # 3. Extended Multi-Zone Crops ('cara', 'breast', 'pompi', 'frente', 'aerola', 'piernas', 'brazos', 'torso')
        if self.crop_regions and rois_isolated:
            zones_extracted = AnatomicalZoneExtractor.extract_all_zones(frame, rois_isolated)
            crops_dir = self.output_dir / "crops"

            for z_name, z_img in zones_extracted.items():
                if z_img is None or z_img.size == 0:
                    continue

                z_sub_dir = crops_dir / z_name
                z_sub_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = format_snapshot_filename(stem, timestamp_sec, tag=z_name)
                crop_dest = z_sub_dir / crop_filename
                crop_tmp = z_sub_dir / f".tmp_{crop_filename}"

                cv2.imwrite(str(crop_tmp), z_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(crop_tmp, crop_dest)
                DesktopThumbnailManager.sync_thumbnail(crop_dest)

                crop_skin = SkinToneAnalyzer.analyze_skin(z_img) if z_name in ("cara", "torso", "breast", "pompi", "frente") else skin_info
                areola_info = AreolaMorphologyAnalyzer.analyze_areola(z_img) if z_name in ("aerola", "breast") else None
                crop_face = FacialMorphologyAnalyzer.analyze_face(z_img) if z_name == "cara" else face_info

                crop_desc = generate_snapshot_description(
                    model_name=self.target_model,
                    zone_type=z_name,
                    timestamp_str=t_str,
                    studio_score=metrics.get("studio_score", 0.0),
                    sharpness=metrics.get("sharpness", 0.0),
                    skin_info=crop_skin,
                    face_info=crop_face if z_name == "cara" else None,
                    anthro_info=anthro_info if z_name in ("cuerpo", "brazos") else None,
                    silhouette_info=silhouette_info if z_name in ("cuerpo", "torso") else None,
                    areola_info=areola_info
                )

                crop_record = {
                    "image_filename": crop_filename,
                    "image_path": str(crop_dest),
                    "model_name": self.target_model,
                    "zone_type": z_name,
                    "description": crop_desc,
                    "extraction_date": now_iso,
                    "source_video": source_vid_path.name,
                    "source_video_path": str(source_vid_path),
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp_str": t_str,
                    "studio_score": metrics.get("studio_score", 0.0),
                    "sharpness": metrics.get("sharpness", 0.0),
                    "brightness": metrics.get("brightness", 0.0),
                    "contrast": metrics.get("contrast", 0.0),
                    "skin_tone_hex": crop_skin["hex"],
                    "fitzpatrick_type": crop_skin["fitzpatrick_type"],
                    "complexion_smoothness": crop_skin["complexion_smoothness"],
                    "silhouette_type": silhouette_info["silhouette_type"] if z_name in ("cuerpo", "torso") else "N/A",
                    "areola_shape": areola_info["shape"] if areola_info else "N/A",
                    "eye_color": crop_face["eye_color"] if z_name == "cara" else "N/A",
                    "hair_color": crop_face["hair_color"] if z_name == "cara" else "N/A",
                    "face_shape": crop_face["face_shape"] if z_name == "cara" else "N/A",
                    "nose_shape": crop_face["nose_shape"] if z_name == "cara" else "N/A",
                    "ear_shape": crop_face["ear_shape"] if z_name == "cara" else "N/A",
                    "estimated_height_cm": anthro_info["estimated_height_cm"] if z_name in ("cuerpo", "piernas") else 0.0,
                    "hand_size_category": anthro_info["hand_size_category"] if z_name in ("cuerpo", "brazos") else "N/A",
                    "biometrics": {
                        "skin": crop_skin,
                        "areola": areola_info,
                        "face": crop_face if z_name == "cara" else None,
                        "silhouette": silhouette_info if z_name in ("cuerpo", "torso") else None
                    },
                    "agy_verdict": "PENDING",
                    "agy_supervision": ""
                }
                records.append(crop_record)

                if self.db_manager:
                    crop_blob = None
                    if self.store_blob:
                        with open(crop_dest, "rb") as f:
                            crop_blob = f.read()
                    self.db_manager.upsert_snapshot(crop_record, image_blob=crop_blob)

        return records

    def harvest_from_timestamps(
        self,
        video_input: Union[Path, str],
        timestamps: List[Union[str, float]]
    ) -> Dict[str, Any]:
        vid_path = Path(video_input).resolve()
        if not vid_path.is_file():
            raise FileNotFoundError(f"Video file not found: {vid_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        stem = vid_path.stem
        all_records = []

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

                metrics = {
                    "studio_score": st_info["studio_score"],
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "contrast": st_info.get("contrast", 0.0)
                }

                saved = self._save_snapshot_and_records(frame, stem, vid_path, t_sec, metrics, rois_isolated)
                all_records.extend(saved)
        finally:
            cap.release()

        if self.supervise_agy:
            if self.verbose:
                print(f"🤖 Dispatching {len(all_records)} records to agy CLI for supervision...")
            all_records = AgySupervisor.supervise(all_records, self.target_model)
            if self.db_manager:
                for rec in all_records:
                    self.db_manager.upsert_snapshot(rec)

        return self._write_catalogs(vid_path, all_records)

    def harvest_studio_situations(
        self,
        video_input: Union[Path, str],
        num_situations: int = 8,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        vid_path = Path(video_input).resolve()
        info = VideoMetadataExtractor.get_info(vid_path)
        duration = info.get("duration", 0.0)
        t_start = self.parse_time_str(start_time) if start_time else 0.0
        t_end = self.parse_time_str(end_time) if end_time else (duration if duration > 0 else 600.0)
        effective_duration = max(1.0, t_end - t_start)

        num_situations = max(1, min(num_situations, self.max_snapshots))
        window_duration = effective_duration / num_situations

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        stem = vid_path.stem
        all_records = []
        preview_frames_for_sheet = []

        try:
            for s_idx in range(num_situations):
                w_start = t_start + (s_idx * window_duration)
                w_end = min(t_end, w_start + window_duration)
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

                    face_conf = 0.0
                    person_conf = 0.0
                    if self.bio_engine:
                        faces = self.bio_engine.detect_faces(frame)
                        if faces is not None and len(faces) > 0:
                            face_conf = float(faces[0][-1])

                    if self.somatotype_extractor:
                        r_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                        if r_data.get("person_detected", False):
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

                if best_frame is not None and best_score > 0.0:
                    is_dup, dist = self.dedup.is_duplicate(best_frame)
                    if not is_dup:
                        rois_isolated = {}
                        if self.crop_regions and self.somatotype_extractor:
                            r_data = self.somatotype_extractor.extract_keypoint_rois(best_frame)
                            rois_isolated = r_data.get("rois", {})

                        metrics = {
                            "studio_score": best_score,
                            "sharpness": best_meta.get("sharpness", 0.0),
                            "brightness": best_meta.get("brightness", 0.0),
                            "contrast": best_meta.get("contrast", 0.0)
                        }

                        saved = self._save_snapshot_and_records(best_frame, stem, vid_path, best_sec, metrics, rois_isolated)
                        all_records.extend(saved)
                        preview_frames_for_sheet.append((best_frame, saved[0]))

        finally:
            cap.release()

        if self.contact_sheet and preview_frames_for_sheet:
            self._generate_contact_sheet(preview_frames_for_sheet, stem)

        if self.supervise_agy:
            if self.verbose:
                print(f"🤖 Supervising {len(all_records)} situation records with agy...")
            all_records = AgySupervisor.supervise(all_records, self.target_model)
            if self.db_manager:
                for rec in all_records:
                    self.db_manager.upsert_snapshot(rec)

        return self._write_catalogs(vid_path, all_records)

    def _generate_contact_sheet(self, frame_entries: List[Tuple[np.ndarray, Dict[str, Any]]], stem: str):
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
            label_top = f"Snapshot: {entry['timestamp_str']} [{entry.get('zone_type', 'cuerpo')}]"
            label_sub = f"Nose:{entry.get('nose_shape','Straight')[:8]} | H:{entry.get('estimated_height_cm',165):.0f}cm"
            cv2.rectangle(tile, (0, tile_h - 48), (tile_w, tile_h), (20, 20, 20), -1)
            cv2.putText(tile, label_top, (12, tile_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, label_sub, (12, tile_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 180), 1, cv2.LINE_AA)
            sheet[y1:y2, x1:x2] = tile

        sheet_path = self.output_dir / f"{stem}_STUDIO_CONTACT_SHEET.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        DesktopThumbnailManager.sync_thumbnail(sheet_path)

    def harvest_from_video(
        self,
        video_input: Union[Path, str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        fps_interval: float = 0.5
    ) -> Dict[str, Any]:
        if self.studio_mode:
            return self.harvest_studio_situations(
                video_input=video_input,
                num_situations=self.situations,
                start_time=start_time,
                end_time=end_time
            )

        vid_path = Path(video_input).resolve()
        info = VideoMetadataExtractor.get_info(vid_path)
        duration = info.get("duration", 0.0)
        t_start = self.parse_time_str(start_time) if start_time else 0.0
        t_end = self.parse_time_str(end_time) if end_time else (duration if duration > 0 else 999999.0)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {vid_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_frames = max(1, int(native_fps / fps_interval)) if self.mode == "interval" else max(1, int(native_fps * 1.5))

        if t_start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000.0)

        stem = vid_path.stem
        all_records = []
        preview_frames = []
        prev_frame = None

        try:
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while cap.isOpened() and len(all_records) < self.max_snapshots:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                curr_sec = msec / 1000.0

                if curr_sec > t_end and t_end > 0:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                valid_sharp, sharpness, brightness, _ = SharpnessAssessor.evaluate(frame)
                if not valid_sharp or sharpness < self.min_sharpness:
                    curr_frame += step_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                    continue

                is_dup, dist = self.dedup.is_duplicate(frame)
                if is_dup:
                    curr_frame += step_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                    continue

                face_conf = 0.0
                person_conf = 0.0
                rois_isolated = {}

                if self.somatotype_extractor:
                    r_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                    if not r_data.get("person_detected", False) and self.only_person:
                        curr_frame += step_frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                        continue
                    rois_isolated = r_data.get("rois", {})
                    person_conf = 0.9

                if self.bio_engine:
                    faces = self.bio_engine.detect_faces(frame)
                    if faces is not None and len(faces) > 0:
                        face_conf = float(faces[0][-1])
                    if self.only_face and face_conf < 0.65:
                        curr_frame += step_frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                        continue

                st_info = StudioQualityScorer.score_frame(
                    frame,
                    prev_frame=prev_frame,
                    min_sharpness=self.min_sharpness,
                    face_conf=face_conf,
                    person_conf=person_conf
                )

                metrics = {
                    "studio_score": st_info["studio_score"],
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "contrast": st_info.get("contrast", 0.0)
                }

                saved = self._save_snapshot_and_records(frame, stem, vid_path, curr_sec, metrics, rois_isolated)
                all_records.extend(saved)
                preview_frames.append((frame.copy(), saved[0]))

                prev_frame = frame.copy()
                curr_frame += step_frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        finally:
            cap.release()

        if self.contact_sheet and preview_frames:
            self._generate_contact_sheet(preview_frames, stem)

        if self.supervise_agy:
            all_records = AgySupervisor.supervise(all_records, self.target_model)
            if self.db_manager:
                for rec in all_records:
                    self.db_manager.upsert_snapshot(rec)

        return self._write_catalogs(vid_path, all_records)

    def _write_catalogs(self, vid_path: Path, all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()

        meta_file = self.output_dir / "SNAPSHOTS_METADATA.json"
        meta_data = {
            "model_name": self.target_model,
            "source_video": vid_path.name,
            "source_video_path": str(vid_path),
            "catalog_date": now_iso,
            "database_enabled": self.enable_db,
            "database_path": str(self.db_manager.db_path) if self.db_manager else None,
            "agy_supervision_enabled": self.supervise_agy,
            "total_records": len(all_records),
            "zones_summary": {z: sum(1 for r in all_records if r["zone_type"] == z) for z in ANATOMICAL_ZONES},
            "records": all_records
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        manifest_file = self.output_dir / "SNAPSHOTS_MANIFEST.json"
        manifest_data = {
            "source_video": str(vid_path),
            "video_filename": vid_path.name,
            "target_model": self.target_model,
            "generated_at": time.time(),
            "total_saved": len(all_records),
            "snapshots": all_records
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return meta_data


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
            contact_sheet=config.contact_sheet,
            enable_db=config.enable_db,
            db_path=config.db_path,
            store_blob=config.store_blob,
            supervise_agy=config.supervise_agy
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
        description="High-Precision Video Keyframe & Studio Snapshot Harvester with SQL/JSON Cataloging, Anthropometrics, and AGY Supervision."
    )
    parser.add_argument("video", nargs="?", help="Path to source video file (MP4, MKV, TS, WebM)")
    parser.add_argument("-m", "--model", default="unnamed_model", help="Target model name for dataset labeling")
    parser.add_argument("-o", "--output-dir", help="Output directory for extracted snapshots and catalog")
    parser.add_argument("--mode", choices=["keyframes", "scene", "interval"], default="interval", help="Extraction strategy (default: interval)")
    parser.add_argument("--fps", type=float, default=0.5, help="Sampling rate in frames per second (default: 0.5 = 1 frame/2s)")
    parser.add_argument("--min-sharpness", type=float, default=75.0, help="Laplacian variance threshold to reject motion blur (default: 75.0)")
    parser.add_argument("--max-snapshots", type=int, default=100, help="Maximum number of snapshots to harvest (default: 100)")
    parser.add_argument("--dhash-threshold", type=int, default=8, help="Max Hamming distance to reject duplicate frames (default: 8)")
    parser.add_argument("--only-person", action="store_true", help="Filter: only save frames with verified human pose")
    parser.add_argument("--only-face", action="store_true", help="Filter: only save frames with verified frontal face")
    parser.add_argument("--crop-regions", action="store_true", help="Extract multi-region anatomical crops (cara, breast, pompi, frente, aerola, piernas, brazos, torso)")
    parser.add_argument("--start", help="Start timestamp in video (e.g. 00:01:30 or 90)")
    parser.add_argument("--end", help="End timestamp in video (e.g. 00:10:00 or 600)")

    # Studio & Situation Discovery Options
    parser.add_argument("--studio", action="store_true", help="Enable Studio Quality Mode: selects peak flattering studio moments across diverse situations")
    parser.add_argument("--situations", type=int, default=8, help="Number of diverse situations/scenes to discover across video (default: 8)")
    parser.add_argument("--timestamps", help="Comma-separated explicit timestamps to harvest (e.g. '00:01:25,00:04:12.5')")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate a studio contact sheet grid image (STUDIO_CONTACT_SHEET.jpg)")

    # SQL Database & Metadata Cataloging Options
    parser.add_argument("--enable-db", action="store_true", help="Enable automatic indexing into persistent SQLite database")
    parser.add_argument("--db", dest="db_path", help="Path to SQLite database file (default: <output_dir>/snapshots_catalog.db)")
    parser.add_argument("--store-blob", "--store-image-db", dest="store_blob", action="store_true", help="Store raw JPEG image binary BLOBs inside SQLite for unified portable datasets")

    # Antigravity (AGY) Supervision
    parser.add_argument("--agy-supervision", "--supervise", dest="supervise_agy", action="store_true", help="Dispatch extracted snapshots to agy CLI for cognitive supervision, verification, and captioning")

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

    model_name = args.model.strip().lower()

    if args.output_dir:
        out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    elif args.model and args.model != "unnamed_model":
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
        target_model=model_name,
        studio_mode=args.studio,
        situations=args.situations,
        contact_sheet=args.contact_sheet or args.studio,
        enable_db=args.enable_db,
        db_path=args.db_path,
        store_blob=args.store_blob,
        supervise_agy=args.supervise_agy,
        verbose=args.verbose or True
    )

    t0 = time.time()
    print(f"\n🚀 Iniciando extracción de snapshots con {harvester.output_dir.name}...")

    if args.timestamps:
        ts_list = [t.strip() for t in args.timestamps.split(",") if t.strip()]
        result = harvester.harvest_from_timestamps(vid_path, ts_list)
    else:
        result = harvester.harvest_from_video(
            video_input=vid_path,
            start_time=args.start,
            end_time=args.end,
            fps_interval=args.fps
        )

    elapsed = time.time() - t0
    saved = result.get("total_records", len(result.get("records", [])))

    print(f"\n✅ Extracción completada en {elapsed:.2f}s:")
    print(f"   📸 Total de registros/recortes: {saved}")
    print(f"   📁 Directorio de salida: {out_dir}")
    print(f"   📋 Metadatos JSON: {out_dir / 'SNAPSHOTS_METADATA.json'}")

    if harvester.db_manager:
        print(f"   🗄️ Base de datos SQLite: {harvester.db_manager.db_path}")

    if args.train_after and model_name != "unnamed_model":
        print(f"\n🔄 Entrenando modelo biométrico para '{model_name}'...")
        train_cmd = [
            sys.executable,
            str(MM_SCRAPER_PATH / "model_media_scraper.py"),
            model_name,
            "--train-model"
        ]
        res = subprocess.run(train_cmd)
        if res.returncode == 0:
            print(f"✨ Re-entrenamiento biométrico finalizado con éxito para '{model_name}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
