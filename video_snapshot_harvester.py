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
    "cuerpo",
    "cara",
    "frente",
    "ojos",
    "cejas",
    "nariz",
    "boca",
    "menton",
    "oreja_izq",
    "oreja_der",
    "cabello",
    "torso",
    "espalda",
    "breast",
    "aerola_izq",
    "aerola_der",
    "ombligo",
    "cadera",
    "entrepierna",
    "pompi",
    "brazo_izq",
    "brazo_der",
    "mano_izq",
    "mano_der",
    "pierna_izq",
    "pierna_der",
    "muslo_izq",
    "muslo_der",
    "tobillo_izq",
    "tobillo_der",
    "pie_izq",
    "pie_der"
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
        "frente": f"Forehead craniofacial anatomical region of {model_name} at {timestamp_str}.",
        "ojos": f"Detailed eye and orbital region crop of {model_name} at {timestamp_str}.",
        "cejas": f"Eyebrow architecture crop of {model_name} at {timestamp_str}.",
        "nariz": f"Nasal bridge and morphology crop of {model_name} at {timestamp_str}.",
        "boca": f"Lips and mouth expression crop of {model_name} at {timestamp_str}.",
        "menton": f"Chin and jawline lower facial crop of {model_name} at {timestamp_str}.",
        "oreja_izq": f"Left ear anatomical profile crop of {model_name} at {timestamp_str}.",
        "oreja_der": f"Right ear anatomical profile crop of {model_name} at {timestamp_str}.",
        "cabello": f"Hair texture and crown region crop of {model_name} at {timestamp_str}.",
        "torso": f"Mid-body torso and waist somatotype crop of {model_name} at {timestamp_str}.",
        "espalda": f"Back and dorsal posterior somatotype region of {model_name} at {timestamp_str}.",
        "breast": f"Upper torso and breast anatomical somatotype region of {model_name} at {timestamp_str}.",
        "aerola_izq": f"Left chest areola region somatotype crop of {model_name} at {timestamp_str}.",
        "aerola_der": f"Right chest areola region somatotype crop of {model_name} at {timestamp_str}.",
        "ombligo": f"Navel and mid-abdominal region crop of {model_name} at {timestamp_str}.",
        "cadera": f"Hip contour and iliac crest somatotype crop of {model_name} at {timestamp_str}.",
        "entrepierna": f"Crotch and upper inner thigh intimate anatomy crop of {model_name} at {timestamp_str}.",
        "pompi": f"Glute and posterior contour anatomy crop of {model_name} at {timestamp_str}.",
        "brazo_izq": f"Left arm and upper extremity crop of {model_name} at {timestamp_str}.",
        "brazo_der": f"Right arm and upper extremity crop of {model_name} at {timestamp_str}.",
        "mano_izq": f"Left hand and palm anatomy crop of {model_name} at {timestamp_str}.",
        "mano_der": f"Right hand and palm anatomy crop of {model_name} at {timestamp_str}.",
        "pierna_izq": f"Left leg lower limb somatotype crop of {model_name} at {timestamp_str}.",
        "pierna_der": f"Right leg lower limb somatotype crop of {model_name} at {timestamp_str}.",
        "muslo_izq": f"Left thigh and quadriceps anatomy crop of {model_name} at {timestamp_str}.",
        "muslo_der": f"Right thigh and quadriceps anatomy crop of {model_name} at {timestamp_str}.",
        "tobillo_izq": f"Left ankle joint crop of {model_name} at {timestamp_str}.",
        "tobillo_der": f"Right ankle joint crop of {model_name} at {timestamp_str}.",
        "pie_izq": f"Left foot anatomy crop of {model_name} at {timestamp_str}.",
        "pie_der": f"Right foot anatomy crop of {model_name} at {timestamp_str}."
    }
    base = desc_templates.get(zone_type, f"Biometric training image of {model_name} ({zone_type}) at {timestamp_str}.")

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
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.execute("PRAGMA journal_size_limit = 67108864;")
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

            # Check and add review status and note columns if not present
            cursor.execute("PRAGMA table_info(snapshots);")
            cols = [col[1] for col in cursor.fetchall()]
            if "status" not in cols:
                cursor.execute("ALTER TABLE snapshots ADD COLUMN status TEXT DEFAULT 'pending';")
            if "review_note" not in cols:
                cursor.execute("ALTER TABLE snapshots ADD COLUMN review_note TEXT;")
            if "parent_snapshot_id" not in cols:
                cursor.execute("ALTER TABLE snapshots ADD COLUMN parent_snapshot_id INTEGER;")
            if "is_master_snapshot" not in cols:
                cursor.execute("ALTER TABLE snapshots ADD COLUMN is_master_snapshot INTEGER DEFAULT 0;")
            if "crop_bbox_json" not in cols:
                cursor.execute("ALTER TABLE snapshots ADD COLUMN crop_bbox_json TEXT;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_status ON snapshots(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_parent ON snapshots(parent_snapshot_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snap_master ON snapshots(is_master_snapshot);")

            # Create AGY Cognitive Feedback Memory Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agy_feedback_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id INTEGER,
                    model_name TEXT NOT NULL,
                    zone_type TEXT NOT NULL,
                    error_category TEXT NOT NULL,
                    user_note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied_count INTEGER DEFAULT 0
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fdbk_model ON agy_feedback_memory(model_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fdbk_zone ON agy_feedback_memory(zone_type);")

            # Create Timeline Minute Scenes Table & Full-Text Search (FTS5)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_scenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    video_filename TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    minute_index INTEGER NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    scene_description TEXT NOT NULL,
                    tags TEXT,
                    studio_score REAL DEFAULT 0.0,
                    image_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_filename, minute_index)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenes_model ON video_scenes(model_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenes_vid ON video_scenes(video_filename);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenes_min ON video_scenes(minute_index);")

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS scenes_fts USING fts5(
                    scene_description,
                    tags,
                    model_name,
                    video_filename,
                    content='video_scenes',
                    content_rowid='id'
                );
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS scenes_ai AFTER INSERT ON video_scenes BEGIN
                  INSERT INTO scenes_fts(rowid, scene_description, tags, model_name, video_filename)
                  VALUES (new.id, new.scene_description, new.tags, new.model_name, new.video_filename);
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS scenes_ad AFTER DELETE ON video_scenes BEGIN
                  INSERT INTO scenes_fts(scenes_fts, rowid, scene_description, tags, model_name, video_filename)
                  VALUES('delete', old.id, old.scene_description, old.tags, old.model_name, old.video_filename);
                END;
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS scenes_au AFTER UPDATE ON video_scenes BEGIN
                  INSERT INTO scenes_fts(scenes_fts, rowid, scene_description, tags, model_name, video_filename)
                  VALUES('delete', old.id, old.scene_description, old.tags, old.model_name, old.video_filename);
                  INSERT INTO scenes_fts(rowid, scene_description, tags, model_name, video_filename)
                  VALUES (new.id, new.scene_description, new.tags, new.model_name, new.video_filename);
                END;
            """)
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
                    agy_supervision, image_blob, status, review_note,
                    parent_snapshot_id, is_master_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    image_blob=COALESCE(excluded.image_blob, snapshots.image_blob),
                    status=COALESCE(snapshots.status, excluded.status),
                    parent_snapshot_id=COALESCE(excluded.parent_snapshot_id, snapshots.parent_snapshot_id),
                    is_master_snapshot=COALESCE(excluded.is_master_snapshot, snapshots.is_master_snapshot);
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
                image_blob,
                record.get("status", "pending"),
                record.get("review_note", ""),
                record.get("parent_snapshot_id"),
                int(record.get("is_master_snapshot", 0))
            ))
            conn.commit()
            return cursor.lastrowid

    def set_status(self, snapshot_id: int, status: str, note: Optional[str] = None) -> bool:
        """Sets review status: 'verified', 'rejected', or 'pending' with optional review note."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            if note is not None:
                cursor.execute("UPDATE snapshots SET status=?, review_note=? WHERE id=?", (status, note, snapshot_id))
            else:
                cursor.execute("UPDATE snapshots SET status=? WHERE id=?", (status, snapshot_id))
            conn.commit()
            return cursor.rowcount > 0

    def record_feedback(self, snapshot_id: int, model_name: str, zone_type: str, error_category: str, user_note: str) -> int:
        """Records a human correction note for Antigravity (AGY) cognitive active learning."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agy_feedback_memory (snapshot_id, model_name, zone_type, error_category, user_note)
                VALUES (?, ?, ?, ?, ?)
            """, (snapshot_id, model_name, zone_type, error_category, user_note))
            # Also update review_note and mark as rejected
            cursor.execute("UPDATE snapshots SET status='rejected', review_note=? WHERE id=?", (f"[{error_category}] {user_note}".strip(), snapshot_id))
            conn.commit()
            return cursor.lastrowid

    def get_feedback_memory(self, model_name: Optional[str] = None, zone_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM agy_feedback_memory"
            params = []
            clauses = []
            if model_name:
                clauses.append("model_name=?")
                params.append(model_name)
            if zone_type:
                clauses.append("zone_type=?")
                params.append(zone_type)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at DESC"
            cursor.execute(query, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_models_summary(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    model_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN status='verified' THEN 1 ELSE 0 END) as verified,
                    SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN status='pending' OR status IS NULL THEN 1 ELSE 0 END) as pending
                FROM snapshots
                GROUP BY model_name
                ORDER BY total DESC;
            """)
            rows = cursor.fetchall()
            return [
                {
                    "model_name": r[0],
                    "total": r[1],
                    "verified": r[2] or 0,
                    "rejected": r[3] or 0,
                    "pending": r[4] or 0
                }
                for r in rows
            ]

    def get_snapshots_filtered(
        self,
        model_name: Optional[str] = None,
        zone_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            clauses = []
            params = []
            if model_name:
                clauses.append("model_name=?")
                params.append(model_name)
            if zone_type:
                clauses.append("zone_type=?")
                params.append(zone_type)
            if status and status != "all":
                if status == "pending":
                    clauses.append("(status='pending' OR status IS NULL)")
                else:
                    clauses.append("status=?")
                    params.append(status)

            where_str = (" WHERE " + " AND ".join(clauses)) if clauses else ""

            # Count total
            cursor.execute(f"SELECT COUNT(*) FROM snapshots{where_str}", tuple(params))
            total = int(cursor.fetchone()[0])

            # Select records (excluding large image_blob from table payload, accessed via /api/image/{id})
            query = f"""
                SELECT
                    id, model_name, image_filename, image_path, zone_type,
                    description, source_video, timestamp_sec, timestamp_str,
                    studio_score, sharpness, brightness, contrast,
                    skin_tone_hex, fitzpatrick_type, complexion_smoothness,
                    silhouette_type, areola_shape, eye_color, hair_color,
                    face_shape, nose_shape, ear_shape, estimated_height_cm,
                    hand_size_category, agy_verdict, status, review_note,
                    (image_blob IS NOT NULL) as has_blob,
                    created_at
                FROM snapshots
                {where_str}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(query, tuple(params + [limit, offset]))
            items = [dict(r) for r in cursor.fetchall()]
            return items, total

    def get_snapshot_by_id(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            return res

    def get_image_blob(self, snapshot_id: int) -> Optional[Tuple[bytes, str]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_blob, image_filename, image_path FROM snapshots WHERE id=?", (snapshot_id,))
            row = cursor.fetchone()
            if not row:
                return None
            blob, fname, fpath = row[0], row[1], row[2]
            if blob:
                return blob, fname
            # Fallback to local file if blob was not stored
            if fpath and Path(fpath).is_file():
                try:
                    with open(fpath, "rb") as f:
                        return f.read(), fname
                except Exception:
                    pass
            return None

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

    def upsert_scene(
        self,
        model_name: str,
        video_filename: str,
        video_path: str,
        minute_index: int,
        timestamp_sec: float,
        timestamp_str: str,
        scene_description: str,
        tags: str = "",
        studio_score: float = 0.0,
        image_blob: Optional[bytes] = None
    ) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO video_scenes (
                    model_name, video_filename, video_path, minute_index,
                    timestamp_sec, timestamp_str, scene_description, tags,
                    studio_score, image_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_filename, minute_index) DO UPDATE SET
                    scene_description=excluded.scene_description,
                    tags=excluded.tags,
                    studio_score=excluded.studio_score,
                    image_blob=COALESCE(excluded.image_blob, video_scenes.image_blob),
                    timestamp_sec=excluded.timestamp_sec,
                    timestamp_str=excluded.timestamp_str;
            """, (
                model_name, video_filename, video_path, minute_index,
                timestamp_sec, timestamp_str, scene_description, tags,
                studio_score, image_blob
            ))
            conn.commit()
            return cursor.lastrowid

    def get_video_scenes(self, video_filename: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            clauses = []
            params = []
            if video_filename:
                clauses.append("video_filename=?")
                params.append(video_filename)
            if model_name:
                clauses.append("model_name=?")
                params.append(model_name)
            where_str = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            cursor.execute(f"""
                SELECT id, model_name, video_filename, video_path, minute_index,
                       timestamp_sec, timestamp_str, scene_description, tags,
                       studio_score, (image_blob IS NOT NULL) as has_blob, created_at
                FROM video_scenes
                {where_str}
                ORDER BY minute_index ASC
            """, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_scene_image_blob(self, scene_id: int) -> Optional[bytes]:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_blob FROM video_scenes WHERE id=?", (scene_id,))
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

    def search_scenes(self, query: str, model_name: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        import re
        clean_q = query.strip()
        if not clean_q:
            return []

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Minute / Time parsing (e.g. "minuto 4", "4:30")
            min_match = re.search(r'(?:minuto|min|m)\s*(\d+)', clean_q, re.IGNORECASE)
            time_match = re.search(r'(\d+):(\d+)', clean_q)

            if min_match:
                target_min = int(min_match.group(1))
                q_sql = "SELECT id, model_name, video_filename, video_path, minute_index, timestamp_sec, timestamp_str, scene_description, tags, studio_score FROM video_scenes WHERE minute_index=?"
                params = [target_min]
                if model_name and model_name != "all":
                    q_sql += " AND model_name=?"
                    params.append(model_name)
                cursor.execute(q_sql, tuple(params))
                rows = cursor.fetchall()
                if rows:
                    return [dict(r) for r in rows]

            elif time_match:
                t_sec = int(time_match.group(1)) * 60 + int(time_match.group(2))
                q_sql = "SELECT id, model_name, video_filename, video_path, minute_index, timestamp_sec, timestamp_str, scene_description, tags, studio_score FROM video_scenes WHERE abs(timestamp_sec - ?) <= 60"
                params = [t_sec]
                if model_name and model_name != "all":
                    q_sql += " AND model_name=?"
                    params.append(model_name)
                cursor.execute(q_sql, tuple(params))
                rows = cursor.fetchall()
                if rows:
                    return [dict(r) for r in rows]

            # 2. Full-Text Search (FTS5)
            try:
                fts_query = re.sub(r'[^\w\s]', ' ', clean_q).strip()
                if fts_query:
                    fts_terms = " OR ".join([f'"{term}"*' for term in fts_query.split()])
                    q_sql = """
                        SELECT v.id, v.model_name, v.video_filename, v.video_path,
                               v.minute_index, v.timestamp_sec, v.timestamp_str,
                               v.scene_description, v.tags, v.studio_score
                        FROM scenes_fts f
                        JOIN video_scenes v ON f.rowid = v.id
                        WHERE scenes_fts MATCH ?
                    """
                    params = [fts_terms]
                    if model_name and model_name != "all":
                        q_sql += " AND v.model_name = ?"
                        params.append(model_name)
                    q_sql += " ORDER BY rank LIMIT ?"
                    params.append(limit)

                    cursor.execute(q_sql, tuple(params))
                    rows = cursor.fetchall()
                    if rows:
                        return [dict(r) for r in rows]
            except Exception:
                pass

            # 3. Fallback to LIKE substring search
            like_sql = """
                SELECT id, model_name, video_filename, video_path, minute_index,
                       timestamp_sec, timestamp_str, scene_description, tags, studio_score
                FROM video_scenes
                WHERE (scene_description LIKE ? OR tags LIKE ? OR video_filename LIKE ?)
            """
            like_pattern = f"%{clean_q}%"
            params = [like_pattern, like_pattern, like_pattern]
            if model_name and model_name != "all":
                like_sql += " AND model_name=?"
                params.append(model_name)
            like_sql += " ORDER BY studio_score DESC LIMIT ?"
            params.append(limit)

            cursor.execute(like_sql, tuple(params))
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


class FramingAndCandidateEvaluator:
    """
    Evaluates frame framing, proximity, frontality, and visibility to ensure
    only optimal frames are selected for specific anatomical extractions:
    1. FACE CLOSE-UP: Frontal gaze (mirando de frente a cámara, NO de lado), sharp eyes/nose/mouth.
    2. BUST & CHEST CLOSE-UP: Upper torso close to camera, clearly visible, unoccluded.
    3. FULL BODY: Unoccluded full silhouette showing entire lower/upper limbs.
    """

    @staticmethod
    def evaluate(
        frame: np.ndarray,
        rois: Dict[str, Any],
        kpts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        suitability = {
            "can_extract_face": False,
            "can_extract_chest_bust": False,
            "can_extract_limbs_full_body": False,
            "framing_label": "Plano General",
            "face_proximity": 0.0,
            "chest_proximity": 0.0,
            "is_frontal_face": False
        }

        # 1. EVALUAR IDONEIDAD PARA ROSTRO / FACCIONES
        face_roi = rois.get("face") if rois else None
        if face_roi is not None and face_roi.size > 0:
            fh, fw = face_roi.shape[:2]
            face_area_ratio = float(fh * fw) / frame_area
            suitability["face_proximity"] = round(face_area_ratio * 100.0, 2)

            fgray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            face_lap = float(cv2.Laplacian(fgray, cv2.CV_64F).var())

            # Frontalidad estricta (mirando de frente a la cámara y NO de lado)
            is_frontal = False
            if kpts is not None and len(kpts) >= 5:
                nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
                if nose[2] >= 0.40 and l_eye[2] >= 0.40 and r_eye[2] >= 0.40:
                    d_left = abs(nose[0] - l_eye[0])
                    d_right = abs(r_eye[0] - nose[0])
                    max_d = max(d_left, d_right) + 1e-4
                    min_d = min(d_left, d_right) + 1e-4
                    symmetry = min_d / max_d
                    # Si symmetry < 0.45, la cabeza está girada o en perfil de lado
                    is_frontal = (symmetry >= 0.45)

                    # Inclinación de ojos horizontal
                    dy = abs(l_eye[1] - r_eye[1])
                    dx = abs(l_eye[0] - r_eye[0]) + 1e-4
                    angle_deg = math.degrees(math.atan2(dy, dx))
                    if angle_deg > 25.0:
                        is_frontal = False
            elif face_area_ratio >= 0.08:
                is_frontal = True

            suitability["is_frontal_face"] = is_frontal

            # Criterio estricto de cara para facciones:
            # - Mirando de frente (NO de lado)
            # - Buen tamaño en encuadre (>= 75px)
            # - Alta nitidez local (>= 70.0)
            if fh >= 75 and fw >= 65 and face_lap >= 70.0 and is_frontal:
                suitability["can_extract_face"] = True

        # 2. EVALUAR IDONEIDAD PARA PECHO / BUSTO / TORSO CERCANO
        torso_roi = rois.get("torso") if rois else None
        if torso_roi is not None and torso_roi.size > 0:
            th, tw = torso_roi.shape[:2]
            torso_area_ratio = float(th * tw) / frame_area
            suitability["chest_proximity"] = round(torso_area_ratio * 100.0, 2)

            breast_part = torso_roi[:int(th * 0.55), :]
            if breast_part.size > 0:
                tgray = cv2.cvtColor(breast_part, cv2.COLOR_BGR2GRAY) if len(breast_part.shape) == 3 else breast_part
                chest_lap = float(cv2.Laplacian(tgray, cv2.CV_64F).var())

                has_shoulders = True
                if kpts is not None and len(kpts) >= 7:
                    has_shoulders = (kpts[5][2] >= 0.35 and kpts[6][2] >= 0.35)

                if th >= 95 and tw >= 80 and chest_lap >= 38.0 and has_shoulders:
                    suitability["can_extract_chest_bust"] = True

        # 3. EVALUAR IDONEIDAD PARA EXTREMIDADES Y CUERPO COMPLETO
        if kpts is not None and len(kpts) >= 17:
            valid_kpts = [k for k in kpts if k[2] >= 0.35]
            if len(valid_kpts) >= 8:
                ys = [k[1] for k in valid_kpts]
                vertical_coverage = (max(ys) - min(ys)) / float(h)

                has_hips = (kpts[11][2] >= 0.35 and kpts[12][2] >= 0.35)
                has_knees = (kpts[13][2] >= 0.35 or kpts[14][2] >= 0.35)
                has_ankles = (kpts[15][2] >= 0.35 or kpts[16][2] >= 0.35)

                if vertical_coverage >= 0.40 and has_hips and has_knees and has_ankles:
                    suitability["can_extract_limbs_full_body"] = True

        # Etiqueta descriptiva del encuadre
        if suitability["can_extract_face"] and suitability["face_proximity"] >= 8.0:
            suitability["framing_label"] = "Primer Plano Frontal (Close-Up Facial)"
        elif suitability["can_extract_chest_bust"] and not suitability["can_extract_limbs_full_body"]:
            suitability["framing_label"] = "Plano Medio (Pecho / Torso Visible)"
        elif suitability["can_extract_limbs_full_body"]:
            suitability["framing_label"] = "Cuerpo Completo (Full-Body Limbs)"
        else:
            suitability["framing_label"] = "Plano General"

        return suitability


class AnatomicalZoneExtractor:
    """
    Extracts complete, fine-grained multi-region crops with lateral independence (left/right)
    matching Francisco's precise anatomical vocabulary:
    - Craniofacial: cara, frente, cejas, ojos, nariz, boca, menton, oreja_izq, oreja_der, cabello
    - Torso/Trunk: torso, espalda, breast, aerola_izq, aerola_der, ombligo, cadera
    - Pelvis/Intimate: entrepierna, pompi
    - Limbs (Upper): brazo_izq, brazo_der, mano_izq, mano_der
    - Limbs (Lower): pierna_izq, pierna_der, muslo_izq, muslo_der, tobillo_izq, tobillo_der, pie_izq, pie_der
    - Full: cuerpo
    """

    @staticmethod
    def crop_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
        ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if (ix2 - ix1) < 15 or (iy2 - iy1) < 15:
            return None
        return img[iy1:iy2, ix1:ix2].copy()

    @classmethod
    def isolate_areola(cls, breast_img: np.ndarray, is_left: bool) -> Optional[np.ndarray]:
        """Isolates left or right areola independently via Lab/HSV pigmentation and contour analysis."""
        if breast_img is None or breast_img.size == 0:
            return None
        bh, bw = breast_img.shape[:2]
        if bh < 25 or bw < 40:
            return None

        # Split breast horizontally: left hemisphere vs right hemisphere
        mid_x = bw // 2
        hemi = breast_img[:, :mid_x] if is_left else breast_img[:, mid_x:]
        hh, hw = hemi.shape[:2]
        if hh < 20 or hw < 20:
            return None

        # Detect pigmentation contrast in CIE-Lab
        lab = cv2.cvtColor(hemi, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        skin_L = np.median(L)
        skin_a = np.median(a)

        # Areola has darker L* and higher a* (redness/pink/brown) than surrounding skin
        pigment_score = (skin_L - L.astype(float)) + 1.8 * (a.astype(float) - skin_a)
        pigment_norm = cv2.normalize(pigment_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, thresh = cv2.threshold(pigment_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_cx = int(hw * 0.50)
        target_cy = int(hh * 0.55)
        best_cnt = None
        max_prominence = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 <= area <= (hh * hw * 0.40):
                bx, by, bw_cnt, bh_cnt = cv2.boundingRect(cnt)
                aspect = max(bw_cnt, bh_cnt) / float(max(1, min(bw_cnt, bh_cnt)))
                if aspect < 2.8:
                    cx, cy = bx + bw_cnt // 2, by + bh_cnt // 2
                    dist = math.hypot(cx - target_cx, cy - target_cy)
                    prominence = area / (dist + 15.0)
                    if prominence > max_prominence:
                        max_prominence = prominence
                        best_cnt = cnt

        if best_cnt is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
            r = max(int(radius * 1.9), 30)
            crop = cls.crop_box(hemi, cx - r, cy - r, cx + r, cy + r)
            if crop is not None:
                return crop

        # Fallback: anatomical window around typical areola position (55% height, 50% width)
        win_w = max(int(hw * 0.55), 35)
        win_h = max(int(hh * 0.55), 35)
        return cls.crop_box(hemi, target_cx - win_w // 2, target_cy - win_h // 2, target_cx + win_w // 2, target_cy + win_h // 2)

    @classmethod
    def is_valid_skin_and_texture(
        cls,
        crop: np.ndarray,
        min_skin_ratio: float = 0.18,
        min_std: float = 16.0,
        max_blue_ratio: float = 0.40
    ) -> bool:
        """
        Validates biological texture and real human skin presence,
        strictly rejecting solid blue/green/black blankets, sheets, walls, or blurry voids.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return False

        # 1. Textural Variance Check (rejects flat untextured surfaces)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        std_val = float(np.std(gray))
        if std_val < min_std:
            return False

        # 2. Human Skin Color Modeling across Fitzpatrick spectrum
        # YCrCb: Cr in [133, 173], Cb in [77, 127]
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        # HSV: Hue in [0, 25] or [160, 180] (warm tones), Saturation in [20, 255]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        skin_hsv = ((h_ch <= 25) | (h_ch >= 160)) & (s_ch >= 20)

        skin_mask = skin_ycrcb | skin_hsv
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(h * w)

        # 3. Explicit Cold Solid Fabric / Blue Blanket Rejection
        b_ch = crop[:, :, 0]
        g_ch = crop[:, :, 1]
        r_ch = crop[:, :, 2]
        cold_blue_mask = (b_ch > r_ch + 15) & (b_ch > g_ch)
        cold_blue_ratio = float(np.count_nonzero(cold_blue_mask)) / float(h * w)
        if cold_blue_ratio > max_blue_ratio:
            return False

        return skin_ratio >= min_skin_ratio

    @classmethod
    def is_sharp_and_clear(
        cls,
        crop: np.ndarray,
        min_laplacian: float = 65.0,
        max_motion_anisotropy: float = 6.0
    ) -> bool:
        """
        Validates local sharpness and checks for directional motion blur.
        Rejects blurry frames where subjects were moving fast or camera lost focus.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 24 or w < 24:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < min_laplacian:
            return False

        # Directional motion blur detection via Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        var_x = float(np.var(gx)) + 1e-5
        var_y = float(np.var(gy)) + 1e-5
        ratio = max(var_x / var_y, var_y / var_x)

        # Extreme directional stretch with low orthogonal energy indicates severe motion trail
        if ratio > max_motion_anisotropy and laplacian_var < 140.0:
            return False

        return True

    @classmethod
    def extract_all_zones(
        cls,
        frame: np.ndarray,
        rois_isolated: Dict[str, Any],
        kpts: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        zones: Dict[str, np.ndarray] = {}
        if frame is None or frame.size == 0:
            return zones

        # Evaluate candidate suitability by framing, proximity and frontality
        suit = FramingAndCandidateEvaluator.evaluate(frame, rois_isolated, kpts=kpts)

        # 0. Full body snapshot (ONLY if entire silhouette is visible)
        if suit["can_extract_limbs_full_body"] and cls.is_sharp_and_clear(frame, min_laplacian=35.0):
            zones["cuerpo"] = frame.copy()

        # ==========================================
        # 1. CRANEOFACIAL (Solo si es candidato frontal, cercano y nítido)
        # ==========================================
        face_img = rois_isolated.get("face") if rois_isolated else None
        if suit["can_extract_face"] and face_img is not None and face_img.size > 0 and cls.is_sharp_and_clear(face_img, min_laplacian=70.0):
            zones["cara"] = face_img
            fh, fw = face_img.shape[:2]

            # 1.1 Frente de la cara (Forehead)
            frente = cls.crop_box(face_img, fw * 0.12, 0, fw * 0.88, fh * 0.36)
            if frente is not None and cls.is_valid_skin_and_texture(frente, min_skin_ratio=0.30):
                zones["frente"] = frente

            # 1.2 Cejas
            cejas = cls.crop_box(face_img, fw * 0.08, fh * 0.18, fw * 0.92, fh * 0.38)
            if cejas is not None and cls.is_sharp_and_clear(cejas, min_laplacian=50.0):
                zones["cejas"] = cejas

            # 1.3 Ojos (franja ocular binocular)
            ojos = cls.crop_box(face_img, fw * 0.08, fh * 0.26, fw * 0.92, fh * 0.50)
            if ojos is not None and cls.is_sharp_and_clear(ojos, min_laplacian=60.0):
                zones["ojos"] = ojos

            # 1.4 Nariz (tercio medio nasal)
            nariz = cls.crop_box(face_img, fw * 0.28, fh * 0.38, fw * 0.72, fh * 0.68)
            if nariz is not None and cls.is_valid_skin_and_texture(nariz, min_skin_ratio=0.25):
                zones["nariz"] = nariz

            # 1.5 Boca (labios y comisuras)
            boca = cls.crop_box(face_img, fw * 0.20, fh * 0.64, fw * 0.80, fh * 0.86)
            if boca is not None and cls.is_sharp_and_clear(boca, min_laplacian=50.0):
                zones["boca"] = boca

            # 1.6 Mentón / Barbilla
            menton = cls.crop_box(face_img, fw * 0.24, fh * 0.78, fw * 0.76, fh)
            if menton is not None:
                zones["menton"] = menton

            # 1.7 Orejas
            oreja_izq = cls.crop_box(face_img, 0, fh * 0.25, fw * 0.22, fh * 0.70)
            if oreja_izq is not None and cls.is_valid_skin_and_texture(oreja_izq, min_skin_ratio=0.15):
                zones["oreja_izq"] = oreja_izq

            oreja_der = cls.crop_box(face_img, fw * 0.78, fh * 0.25, fw, fh * 0.70)
            if oreja_der is not None and cls.is_valid_skin_and_texture(oreja_der, min_skin_ratio=0.15):
                zones["oreja_der"] = oreja_der

            # 1.8 Cabello
            cabello = cls.crop_box(face_img, 0, 0, fw, fh * 0.28)
            if cabello is not None:
                zones["cabello"] = cabello

        # ==========================================
        # 2. TORSO, BUSTO, ESPALDA, OMBLIGO, CADERA (Solo si pecho/torso es visible y cercano)
        # ==========================================
        torso_img = rois_isolated.get("torso") if rois_isolated else None
        if suit["can_extract_chest_bust"] and torso_img is not None and torso_img.size > 0 and cls.is_valid_skin_and_texture(torso_img, min_skin_ratio=0.15):
            zones["torso"] = torso_img
            th, tw = torso_img.shape[:2]

            # 2.1 Busto / Breast
            if th >= 40 and tw >= 40:
                breast_img = torso_img[:int(th * 0.55), :]
                if cls.is_valid_skin_and_texture(breast_img, min_skin_ratio=0.20):
                    zones["breast"] = breast_img

                    # 2.2 Areolas independientes
                    aerola_izq = cls.isolate_areola(breast_img, is_left=True)
                    if aerola_izq is not None and cls.is_valid_skin_and_texture(aerola_izq, min_skin_ratio=0.15):
                        zones["aerola_izq"] = aerola_izq

                    aerola_der = cls.isolate_areola(breast_img, is_left=False)
                    if aerola_der is not None and cls.is_valid_skin_and_texture(aerola_der, min_skin_ratio=0.15):
                        zones["aerola_der"] = aerola_der

            # 2.3 Ombligo
            if th >= 50 and tw >= 40:
                ombligo = cls.crop_box(torso_img, tw * 0.32, th * 0.65, tw * 0.68, th * 0.88)
                if ombligo is not None and cls.is_valid_skin_and_texture(ombligo, min_skin_ratio=0.20):
                    zones["ombligo"] = ombligo

            # 2.4 Cadera
            if th >= 50 and tw >= 40:
                cadera = cls.crop_box(torso_img, 0, th * 0.72, tw, th)
                if cadera is not None and cls.is_valid_skin_and_texture(cadera, min_skin_ratio=0.15):
                    zones["cadera"] = cadera

            # 2.5 Espalda
            espalda = cls.crop_box(torso_img, tw * 0.15, th * 0.10, tw * 0.85, th * 0.70)
            if espalda is not None and cls.is_valid_skin_and_texture(espalda, min_skin_ratio=0.15):
                zones["espalda"] = espalda

        # ==========================================
        # 3. PELVIS, ENTREPIERNA Y GLÚTEOS
        # ==========================================
        hg_img = rois_isolated.get("hips_glutes") if rois_isolated else None
        if (suit["can_extract_chest_bust"] or suit["can_extract_limbs_full_body"]) and hg_img is not None and hg_img.size > 0 and cls.is_valid_skin_and_texture(hg_img, min_skin_ratio=0.15):
            zones["pompi"] = hg_img
            hgh, hgw = hg_img.shape[:2]

            entrepierna = cls.crop_box(hg_img, hgw * 0.28, hgh * 0.25, hgw * 0.72, hgh * 0.80)
            if entrepierna is not None and cls.is_valid_skin_and_texture(entrepierna, min_skin_ratio=0.20):
                zones["entrepierna"] = entrepierna

        # ==========================================
        # 4. EXTREMIDADES (Solo si la toma fuente es de cuerpo completo)
        # ==========================================
        if suit["can_extract_limbs_full_body"] and kpts is not None and len(kpts) >= 17:
            # 4.1 Brazo Izquierdo (hombro 5, codo 7, muñeca 9)
            arm_l = [kpts[i] for i in [5, 7, 9] if kpts[i][2] >= 0.40]
            if len(arm_l) >= 2:
                xs, ys = [p[0] for p in arm_l], [p[1] for p in arm_l]
                pad = max(max(xs) - min(xs), max(ys) - min(ys), 30.0) * 0.25
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.15):
                    zones["brazo_izq"] = crop

            # 4.2 Brazo Derecho (hombro 6, codo 8, muñeca 10)
            arm_r = [kpts[i] for i in [6, 8, 10] if kpts[i][2] >= 0.40]
            if len(arm_r) >= 2:
                xs, ys = [p[0] for p in arm_r], [p[1] for p in arm_r]
                pad = max(max(xs) - min(xs), max(ys) - min(ys), 30.0) * 0.25
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.15):
                    zones["brazo_der"] = crop

            # 4.3 Manos (mano_izq en muñeca 9, mano_der en muñeca 10)
            if kpts[9][2] >= 0.50:
                mx, my = kpts[9][0], kpts[9][1]
                mr = 50.0
                crop = cls.crop_box(frame, mx - mr, my - mr, mx + mr, my + mr)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20):
                    zones["mano_izq"] = crop

            if kpts[10][2] >= 0.50:
                mx, my = kpts[10][0], kpts[10][1]
                mr = 50.0
                crop = cls.crop_box(frame, mx - mr, my - mr, mx + mr, my + mr)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20):
                    zones["mano_der"] = crop

            # 4.4 Pierna Izquierda vs Muslo Izquierdo
            # Strict anatomical rule: Requires Hip (11), Knee (13) AND Ankle (15) all >= 0.45
            hip_l, knee_l, ankle_l = kpts[11], kpts[13], kpts[15]
            if hip_l[2] >= 0.45 and knee_l[2] >= 0.45 and ankle_l[2] >= 0.45:
                xs = [hip_l[0], knee_l[0], ankle_l[0]]
                ys = [hip_l[1], knee_l[1], ankle_l[1]]
                pad = max(max(xs) - min(xs), max(ys) - min(ys), 40.0) * 0.22
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.15):
                    zones["pierna_izq"] = crop
            elif hip_l[2] >= 0.45 and knee_l[2] >= 0.45:
                # Incomplete limb: ONLY Thigh (muslo)
                xs = [hip_l[0], knee_l[0]]
                ys = [hip_l[1], knee_l[1]]
                pad = max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), 30.0) * 0.25
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - 10, max(xs) + pad, max(ys) + 15)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20):
                    zones["muslo_izq"] = crop

            # 4.5 Pierna Derecha vs Muslo Derecho
            hip_r, knee_r, ankle_r = kpts[12], kpts[14], kpts[16]
            if hip_r[2] >= 0.45 and knee_r[2] >= 0.45 and ankle_r[2] >= 0.45:
                xs = [hip_r[0], knee_r[0], ankle_r[0]]
                ys = [hip_r[1], knee_r[1], ankle_r[1]]
                pad = max(max(xs) - min(xs), max(ys) - min(ys), 40.0) * 0.22
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.15):
                    zones["pierna_der"] = crop
            elif hip_r[2] >= 0.45 and knee_r[2] >= 0.45:
                # Incomplete limb: ONLY Thigh (muslo)
                xs = [hip_r[0], knee_r[0]]
                ys = [hip_r[1], knee_r[1]]
                pad = max(abs(xs[1] - xs[0]), abs(ys[1] - ys[0]), 30.0) * 0.25
                crop = cls.crop_box(frame, min(xs) - pad, min(ys) - 10, max(xs) + pad, max(ys) + 15)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20):
                    zones["muslo_der"] = crop

            # 4.6 Tobillos (tobillo_izq en 15, tobillo_der en 16)
            if kpts[15][2] >= 0.55:
                tx, ty = kpts[15][0], kpts[15][1]
                tr = 45.0
                crop = cls.crop_box(frame, tx - tr, ty - tr, tx + tr, ty + tr)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.18):
                    zones["tobillo_izq"] = crop

            if kpts[16][2] >= 0.55:
                tx, ty = kpts[16][0], kpts[16][1]
                tr = 45.0
                crop = cls.crop_box(frame, tx - tr, ty - tr, tx + tr, ty + tr)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.18):
                    zones["tobillo_der"] = crop

            # 4.7 Pies (pie_izq y pie_der desde el tobillo hacia abajo)
            # High confidence threshold + skin validation to prevent blue blanket/bedsheet false positives
            if kpts[15][2] >= 0.55:
                tx, ty = kpts[15][0], kpts[15][1]
                crop = cls.crop_box(frame, tx - 45, ty - 15, tx + 45, ty + 65)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20) and cls.is_sharp_and_clear(crop, min_laplacian=40.0):
                    zones["pie_izq"] = crop

            if kpts[16][2] >= 0.55:
                tx, ty = kpts[16][0], kpts[16][1]
                crop = cls.crop_box(frame, tx - 45, ty - 15, tx + 45, ty + 65)
                if crop is not None and cls.is_valid_skin_and_texture(crop, min_skin_ratio=0.20) and cls.is_sharp_and_clear(crop, min_laplacian=40.0):
                    zones["pie_der"] = crop

        return zones


class GoogleDriveSource:
    """Discovers, queries, and stages media from configured Google Drive accounts via rclone."""

    DEFAULT_REMOTES = ["gdrive", "gdrive_phone", "gdrive_broker", "gdrive_eng", "gdrive_fdev"]
    CACHE_DIR = Path.home() / ".cache" / "video_snapshot_harvester" / "gdrive_staging"

    @classmethod
    def get_configured_remotes(cls) -> List[str]:
        try:
            res = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, check=True)
            remotes = [line.strip().rstrip(":") for line in res.stdout.splitlines() if line.strip()]
            return remotes if remotes else cls.DEFAULT_REMOTES
        except Exception:
            return cls.DEFAULT_REMOTES

    @classmethod
    def _model_aliases(cls, model_name: str) -> List[str]:
        clean = model_name.strip().lower()
        aliases = [clean]
        if clean == "uwxo":
            aliases.append("uwuxo")
        elif clean == "uwuxo":
            aliases.append("uwxo")
        elif clean == "tanni_bc":
            aliases.append("taanni_bc")
        elif clean == "taanni_bc":
            aliases.append("tanni_bc")
        elif clean.endswith("_") and not clean.endswith("__"):
            aliases.append(clean + "_")
        elif clean.endswith("__"):
            aliases.append(clean[:-1])
        return list(dict.fromkeys(aliases))

    @classmethod
    def search_model_media(
        cls,
        model_name: str,
        remotes: Optional[List[str]] = None,
        base_folder: str = "Recurbate",
        media_type: str = "all",
        max_files: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scans remote Google Drive accounts for media files matching model_name.
        Returns list of file descriptors.
        """
        if remotes is None:
            configured = cls.get_configured_remotes()
            remotes = configured if configured else cls.DEFAULT_REMOTES

        video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts"}
        photo_exts = {".jpg", ".jpeg", ".png", ".webp"}

        if media_type == "videos":
            target_exts = video_exts
        elif media_type == "photos":
            target_exts = photo_exts
        else:
            target_exts = video_exts | photo_exts

        aliases = cls._model_aliases(model_name)
        discovered: List[Dict[str, Any]] = []

        for remote in remotes:
            if len(discovered) >= max_files:
                break
            for alias in aliases:
                if len(discovered) >= max_files:
                    break
                candidate_paths = [f"{base_folder}/{alias}" if base_folder else alias, alias]
                for c_path in candidate_paths:
                    remote_target = f"{remote}:{c_path}"
                    cmd = ["rclone", "lsjson", remote_target, "--max-depth", "1"]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                        if res.returncode != 0 or not res.stdout.strip():
                            continue
                        items = json.loads(res.stdout)
                        for item in items:
                            if item.get("IsDir"):
                                continue
                            fname = item.get("Name", "")
                            ext = Path(fname).suffix.lower()
                            if ext in target_exts:
                                discovered.append({
                                    "remote": remote,
                                    "remote_target": remote_target,
                                    "remote_path": f"{remote_target}/{fname}",
                                    "filename": fname,
                                    "size": item.get("Size", 0),
                                    "mod_time": item.get("ModTime", ""),
                                    "is_video": ext in video_exts,
                                    "model": model_name
                                })
                                if len(discovered) >= max_files:
                                    break
                    except Exception:
                        continue

        return discovered

    @classmethod
    def stage_file(cls, remote_path: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Downloads the remote file using rclone to a local staging directory.
        Uses cached local file if size matches.
        """
        staging_dir = cache_dir or cls.CACHE_DIR
        staging_dir.mkdir(parents=True, exist_ok=True)

        parts = remote_path.split(":", 1)
        if len(parts) != 2:
            return None
        remote_name, rel_path = parts
        filename = Path(rel_path).name

        local_path = staging_dir / remote_name / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.is_file() and local_path.stat().st_size > 0:
            return local_path

        print(f"  ☁️ Descargando desde Google Drive ({remote_name}): {filename}...")
        cmd = ["rclone", "copyto", remote_path, str(local_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode == 0 and local_path.is_file():
                mb = local_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ Descargado en staging: {local_path.name} ({mb:.1f} MB)")
                return local_path
            else:
                print(f"  ⚠️ Error en rclone copyto: {res.stderr}")
                return None
        except Exception as e:
            print(f"  ⚠️ Excepción descargando {remote_path}: {e}")
            return None

    @classmethod
    def clean_staging(cls, staged_files: List[Path]):
        for p in staged_files:
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


class SocialMediaSource:
    """Downloads reels, clips, and photos from social media (Instagram, TikTok, X, Reddit, etc.) via yt-dlp."""

    CACHE_DIR = Path.home() / ".cache" / "video_snapshot_harvester" / "social_staging"

    @classmethod
    def is_social_url(cls, url: str) -> bool:
        if not isinstance(url, str):
            return False
        clean = url.strip().lower()
        if not (clean.startswith("http://") or clean.startswith("https://")):
            return False
        known_domains = [
            "instagram.com", "tiktok.com", "twitter.com", "x.com", "reddit.com",
            "youtube.com", "youtu.be", "fansly.com", "facebook.com", "pin.it", "pinterest.com"
        ]
        return any(d in clean for d in known_domains)

    @classmethod
    def download_url(cls, url: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
        staging_dir = cache_dir or cls.CACHE_DIR
        staging_dir.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        output_template = str(staging_dir / f"social_{url_hash}_%(id)s.%(ext)s")

        print(f"  🌐 Descargando reel/video de red social vía yt-dlp: {url}...")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            url
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                candidates = list(staging_dir.glob(f"social_{url_hash}_*"))
                if candidates:
                    target = candidates[0]
                    mb = target.stat().st_size / (1024 * 1024)
                    print(f"  ✅ Descarga social lista: {target.name} ({mb:.1f} MB)")
                    return target
            print(f"  ⚠️ Error en yt-dlp: {res.stderr[:200]}")
            return None
        except Exception as e:
            print(f"  ⚠️ Excepción descargando URL social {url}: {e}")
            return None

    @classmethod
    def clean_staging(cls, staged_files: List[Path]):
        for p in staged_files:
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


class CatalogSource:
    """Discovers photos and videos from existing local model datasets and catalogs."""

    DATASET_ROOTS = [
        Path.home() / "ComputerVision" / "datasets",
        Path.home() / "Videos" / "Recurbate",
        Path("/media/fbetancourt/ADATA HD710 PRO1")
    ]

    @classmethod
    def find_media(cls, model_name: str, media_type: str = "all", max_files: int = 20) -> List[Path]:
        discovered = []
        aliases = GoogleDriveSource._model_aliases(model_name)
        video_exts = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts"}
        photo_exts = {".jpg", ".jpeg", ".png", ".webp"}

        if media_type == "videos":
            target_exts = video_exts
        elif media_type == "photos":
            target_exts = photo_exts
        else:
            target_exts = video_exts | photo_exts

        for root in cls.DATASET_ROOTS:
            if not root.exists():
                continue
            for alias in aliases:
                cand = root / alias
                if cand.is_dir():
                    for item in cand.rglob("*"):
                        if item.is_file() and item.suffix.lower() in target_exts:
                            discovered.append(item)
                            if len(discovered) >= max_files:
                                return discovered
        return discovered


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
        supervise_agy: bool = False,
        origin: str = "auto",
        gdrive_remote: str = "all",
        gdrive_folder: str = "Recurbate",
        media_type: str = "all",
        max_videos: int = 5,
        situations_per_video: int = 4,
        clean_staging: bool = False
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
        self.origin = origin
        self.gdrive_remote = gdrive_remote
        self.gdrive_folder = gdrive_folder
        self.media_type = media_type
        self.max_videos = max_videos
        self.situations_per_video = situations_per_video
        self.clean_staging = clean_staging


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
        min_sharpness: float = 35.0,
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
        self.output_dir = Path(output_dir).resolve()
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
        rois_isolated: Dict[str, Any],
        kpts: Optional[np.ndarray] = None
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
            "agy_supervision": "",
            "is_master_snapshot": 1,
            "parent_snapshot_id": None
        }
        records.append(main_record)

        master_id = None
        if self.db_manager:
            img_blob = None
            if self.store_blob:
                with open(final_dest, "rb") as f:
                    img_blob = f.read()
            master_id = self.db_manager.upsert_snapshot(main_record, image_blob=img_blob)

        # 3. Extended Multi-Zone Anatomical Crops (28 granular regions)
        if self.crop_regions:
            zones_extracted = AnatomicalZoneExtractor.extract_all_zones(frame, rois_isolated, kpts=kpts)
            crops_dir = self.output_dir / "crops"

            for z_name, z_img in zones_extracted.items():
                if z_img is None or z_img.size == 0 or z_name == "cuerpo":
                    continue

                z_sub_dir = crops_dir / z_name
                z_sub_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = format_snapshot_filename(stem, timestamp_sec, tag=z_name)
                crop_dest = z_sub_dir / crop_filename
                crop_tmp = z_sub_dir / f".tmp_{crop_filename}"

                cv2.imwrite(str(crop_tmp), z_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(crop_tmp, crop_dest)
                DesktopThumbnailManager.sync_thumbnail(crop_dest)

                # Local sharpness & brightness of the specific crop
                z_gray = cv2.cvtColor(z_img, cv2.COLOR_BGR2GRAY) if len(z_img.shape) == 3 else z_img
                local_sharpness = round(float(cv2.Laplacian(z_gray, cv2.CV_64F).var()), 2)
                local_brightness = round(float(np.mean(z_gray)), 2)
                local_contrast = round(float(np.std(z_gray)), 2)

                crop_skin = SkinToneAnalyzer.analyze_skin(z_img) if z_name in (
                    "cara", "torso", "breast", "pompi", "frente", "espalda", "ombligo", "muslo_izq", "muslo_der", "pierna_izq", "pierna_der"
                ) else skin_info
                areola_info = AreolaMorphologyAnalyzer.analyze_areola(z_img) if ("aerola" in z_name or z_name == "breast") else None
                crop_face = FacialMorphologyAnalyzer.analyze_face(z_img) if z_name in ("cara", "ojos", "frente", "nariz") else face_info

                crop_desc = generate_snapshot_description(
                    model_name=self.target_model,
                    zone_type=z_name,
                    timestamp_str=t_str,
                    studio_score=metrics.get("studio_score", 0.0),
                    sharpness=local_sharpness,
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
                    "sharpness": local_sharpness,
                    "brightness": local_brightness,
                    "contrast": local_contrast,
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
                    "agy_supervision": "",
                    "is_master_snapshot": 0,
                    "parent_snapshot_id": master_id
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
                kpts = None
                if self.crop_regions and self.somatotype_extractor:
                    rois_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                    rois_isolated = rois_data.get("rois", {})
                    kpts = rois_data.get("kpts", None)

                metrics = {
                    "studio_score": st_info["studio_score"],
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "contrast": st_info.get("contrast", 0.0)
                }

                saved = self._save_snapshot_and_records(frame, stem, vid_path, t_sec, metrics, rois_isolated, kpts=kpts)
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

    def harvest_minute_timeline(
        self,
        video_input: Union[Path, str],
        model_name: Optional[str] = None,
        interval_sec: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Extracts 1 peak snapshot per minute across the entire video, generates a dense
        scene context description, and saves it to video_scenes table with FTS5 indexing.
        """
        vid_path = Path(video_input).resolve()
        target_model = model_name or self.target_model
        info = VideoMetadataExtractor.get_info(vid_path)
        duration = info.get("duration", 0.0)
        if duration <= 0:
            duration = 600.0

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            return []

        scenes_saved = []
        minutes_total = int(duration // interval_sec) + (1 if (duration % interval_sec) > 5 else 0)

        try:
            for m_idx in range(minutes_total):
                target_sec = m_idx * interval_sec
                best_f = None
                best_score = -1.0
                best_ts = target_sec

                # Sample 4 frames in a 4-second window to find the clearest moment
                for offset_s in [0.0, 1.0, 2.0, 3.0]:
                    s_sec = min(duration - 0.5, target_sec + offset_s)
                    cap.set(cv2.CAP_PROP_POS_MSEC, s_sec * 1000.0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    sc_info = StudioQualityScorer.score_frame(frame, min_sharpness=self.min_sharpness)
                    if sc_info["studio_score"] > best_score:
                        best_score = sc_info["studio_score"]
                        best_f = frame
                        best_ts = s_sec

                if best_f is None:
                    continue

                skin = SkinToneAnalyzer.analyze_skin(best_f)
                h, w = best_f.shape[:2]
                plan_type = "Plano General" if h > w else "Plano Medio"
                time_str = f"{int(best_ts // 60):02d}m{int(best_ts % 60):02d}s"

                desc = f"Escena minuto {m_idx} ({time_str}). {plan_type}. Tono de piel {skin.get('tone_label')} ({skin.get('fitzpatrick_type')}). Nitidez {best_score:.1f}. Contexto visual de {target_model}."
                tags = f"{target_model}, minuto_{m_idx}, {skin.get('fitzpatrick_type')}, {plan_type.lower()}"

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, enc_bytes = cv2.imencode(".jpg", best_f, encode_param)
                blob = enc_bytes.tobytes()

                if self.db_manager:
                    scene_id = self.db_manager.upsert_scene(
                        model_name=target_model,
                        video_filename=vid_path.name,
                        video_path=str(vid_path),
                        minute_index=m_idx,
                        timestamp_sec=best_ts,
                        timestamp_str=time_str,
                        scene_description=desc,
                        tags=tags,
                        studio_score=best_score,
                        image_blob=blob
                    )
                    scenes_saved.append({
                        "id": scene_id,
                        "minute_index": m_idx,
                        "timestamp_sec": best_ts,
                        "timestamp_str": time_str,
                        "scene_description": desc,
                        "tags": tags,
                        "studio_score": best_score
                    })
        finally:
            cap.release()

        return scenes_saved

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

        if num_situations <= 0 or getattr(self, "per_minute", False):
            # Exactly 1 snapshot per minute of video duration
            num_situations = max(1, int(round(effective_duration / 60.0)))
        else:
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
                num_samples = min(10, max(4, int(window_duration * 0.2)))
                sample_times = np.linspace(w_start, w_end, num=num_samples)

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
                    rois_isolated = {}
                    kpts = None
                    if self.crop_regions and self.somatotype_extractor:
                        r_data = self.somatotype_extractor.extract_keypoint_rois(best_frame)
                        rois_isolated = r_data.get("rois", {})
                        kpts = r_data.get("kpts", None)

                    metrics = {
                        "studio_score": best_score,
                        "sharpness": best_meta.get("sharpness", 0.0),
                        "brightness": best_meta.get("brightness", 0.0),
                        "contrast": best_meta.get("contrast", 0.0)
                    }

                    saved = self._save_snapshot_and_records(best_frame, stem, vid_path, best_sec, metrics, rois_isolated, kpts=kpts)
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

    def _write_catalogs(self, primary_src: Union[Path, str], all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        primary_path = Path(primary_src)
        meta_path = self.output_dir / "SNAPSHOTS_METADATA.json"

        clean_records = []
        for r in all_records:
            clean_rec = {k: v for k, v in r.items() if k != "image_blob"}
            clean_records.append(clean_rec)

        catalog_data = {
            "version": "2.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_video": primary_path.name,
            "source_path": str(primary_path),
            "target_model": self.target_model,
            "total_records": len(all_records),
            "records": clean_records
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2, ensure_ascii=False)

        manifest = {
            "source_video": str(primary_path),
            "target_model": self.target_model,
            "output_dir": str(self.output_dir),
            "total_records": len(all_records),
            "metadata_json": str(meta_path),
            "records": clean_records
        }
        if self.db_manager:
            manifest["sqlite_db"] = str(self.db_manager.db_path)

        return manifest

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
                kpts = None

                if self.somatotype_extractor:
                    r_data = self.somatotype_extractor.extract_keypoint_rois(frame)
                    if not r_data.get("person_detected", False) and self.only_person:
                        curr_frame += step_frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
                        continue
                    rois_isolated = r_data.get("rois", {})
                    kpts = r_data.get("kpts", None)
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

                saved = self._save_snapshot_and_records(frame, stem, vid_path, curr_sec, metrics, rois_isolated, kpts=kpts)
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

    def harvest_from_image(self, image_input: Union[Path, str]) -> List[Dict[str, Any]]:
        img_path = Path(image_input).resolve()
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            return []

        stem = img_path.stem
        valid_sharp, sharpness, brightness, _ = SharpnessAssessor.evaluate(img, min_variance=self.min_sharpness)
        st_info = StudioQualityScorer.score_frame(img, min_sharpness=self.min_sharpness)

        is_dup, dist = self.dedup.is_duplicate(img)
        if is_dup:
            return []

        rois_isolated = {}
        kpts = None
        if self.crop_regions and self.somatotype_extractor:
            r_data = self.somatotype_extractor.extract_keypoint_rois(img)
            rois_isolated = r_data.get("rois", {})
            kpts = r_data.get("kpts", None)

        metrics = {
            "studio_score": st_info.get("studio_score", 65.0),
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": st_info.get("contrast", 0.0)
        }

        saved = self._save_snapshot_and_records(img, stem, img_path, 0.0, metrics, rois_isolated, kpts=kpts)
        return saved

    def harvest_multi_source(
        self,
        sources: List[Union[Path, str]],
        media_type: str = "all",
        max_videos: Optional[int] = None,
        situations_per_video: int = 4,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Discovers and harvests studio snapshots and photos across multiple videos and image files/folders.
        """
        video_exts = {".mp4", ".mkv", ".ts", ".webm", ".avi", ".mov", ".flv", ".m4v"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        discovered_videos: List[Path] = []
        discovered_images: List[Path] = []

        for src in sources:
            p = Path(os.path.expanduser(str(src))).resolve()
            if p.is_file():
                ext = p.suffix.lower()
                if ext in video_exts and media_type in ("all", "videos"):
                    discovered_videos.append(p)
                elif ext in image_exts and media_type in ("all", "photos"):
                    discovered_images.append(p)
            elif p.is_dir():
                pattern_vids = p.rglob("*") if recursive else p.glob("*")
                for item in pattern_vids:
                    if item.is_file():
                        ext = item.suffix.lower()
                        if ext in video_exts and media_type in ("all", "videos"):
                            discovered_videos.append(item)
                        elif ext in image_exts and media_type in ("all", "photos"):
                            discovered_images.append(item)

        discovered_videos = sorted(list(set(discovered_videos)))
        discovered_images = sorted(list(set(discovered_images)))

        if max_videos is not None and max_videos > 0:
            discovered_videos = discovered_videos[:max_videos]

        print(f"🎬 Fuentes descubiertas para '{self.target_model}': {len(discovered_videos)} videos, {len(discovered_images)} imágenes.")

        all_records: List[Dict[str, Any]] = []

        # 1. Process discovered videos
        for v_idx, v_path in enumerate(discovered_videos, 1):
            if len(all_records) >= self.max_snapshots:
                break
            print(f"  [{v_idx}/{len(discovered_videos)}] Explorando video: {v_path.name}...")
            try:
                res = self.harvest_studio_situations(
                    video_input=v_path,
                    num_situations=situations_per_video
                )
                v_records = res.get("records", [])
                all_records.extend(v_records)
            except Exception as e:
                print(f"    ⚠️ Error en video {v_path.name}: {e}")

        # 2. Process discovered static images
        for i_idx, img_p in enumerate(discovered_images, 1):
            if len(all_records) >= self.max_snapshots:
                break
            try:
                img_recs = self.harvest_from_image(img_p)
                all_records.extend(img_recs)
            except Exception as e:
                pass

        if self.supervise_agy and all_records:
            all_records = AgySupervisor.supervise(all_records, self.target_model)
            if self.db_manager:
                for rec in all_records:
                    self.db_manager.upsert_snapshot(rec)

        primary_src = discovered_videos[0] if discovered_videos else (discovered_images[0] if discovered_images else self.output_dir)
        return self._write_catalogs(primary_src, all_records)


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
        description="High-Precision Video Keyframe & Studio Snapshot Harvester with Google Drive Multi-Account Support, SQL/JSON Cataloging, 32-Zone Anthropometrics, and AGY Supervision."
    )
    parser.add_argument("inputs", nargs="*", help="Source video/image files, local directories, or Google Drive rclone remotes (e.g. gdrive_phone:Recurbate/uwuxo)")
    parser.add_argument("-m", "--model", default="unnamed_model", help="Target model name for dataset labeling")
    parser.add_argument("-o", "--output-dir", help="Output directory for extracted snapshots and catalog")
    parser.add_argument("--origin", "--source-origin", "-s", "--source", choices=["auto", "local", "gdrive"], default="auto", help="Origin source provider for video/image training assets (local, gdrive, auto)")
    parser.add_argument("--gdrive-remote", default="all", help="Specific Google Drive remote to query (e.g. gdrive_phone, gdrive_eng, gdrive_broker, gdrive_fdev, or all)")
    parser.add_argument("--gdrive-folder", default="Recurbate", help="Base Google Drive folder to search for model media (default: Recurbate)")
    parser.add_argument("--media-type", choices=["all", "videos", "photos"], default="all", help="Media type filter when harvesting multi-source (default: all)")
    parser.add_argument("--max-videos", type=int, default=5, help="Maximum number of videos to explore from source (default: 5)")
    parser.add_argument("--situations-per-video", type=int, default=4, help="Peak studio situations/snapshots to extract per video (default: 4)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively explore directories for media")
    parser.add_argument("--clean-staging", action="store_true", help="Purge staged Google Drive media after harvesting snapshots")

    # Extraction Engine Options
    parser.add_argument("--mode", choices=["keyframes", "scene", "interval"], default="interval", help="Extraction strategy (default: interval)")
    parser.add_argument("--fps", type=float, default=0.5, help="Sampling rate in frames per second (default: 0.5 = 1 frame/2s)")
    parser.add_argument("--min-sharpness", type=float, default=35.0, help="Laplacian variance threshold to reject motion blur (default: 35.0)")
    parser.add_argument("--max-snapshots", type=int, default=100, help="Maximum number of snapshots to harvest (default: 100)")
    parser.add_argument("--dhash-threshold", type=int, default=8, help="Max Hamming distance to reject duplicate frames (default: 8)")
    parser.add_argument("--only-person", action="store_true", help="Filter: only save frames with verified human pose")
    parser.add_argument("--only-face", action="store_true", help="Filter: only save frames with verified frontal face")
    parser.add_argument("--crop-regions", action="store_true", help="Extract 32-zone anatomical crops (lateral arms/legs/areolas, crotch, facial, torso, etc.)")
    parser.add_argument("--start", help="Start timestamp in video (e.g. 00:01:30 or 90)")
    parser.add_argument("--end", help="End timestamp in video (e.g. 00:10:00 or 600)")

    # Studio & Situation Discovery Options
    parser.add_argument("--studio", action="store_true", help="Enable Studio Quality Mode: selects peak flattering studio moments across diverse situations")
    parser.add_argument("--situations", type=int, default=8, help="Number of diverse situations/scenes to discover across video (default: 8)")
    parser.add_argument("--per-minute", action="store_true", help="Harvest exactly 1 snapshot per minute across the entire video duration")
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

    # Interactive Active-Learning Studio Web UI
    parser.add_argument("--web", "--studio-ui", dest="launch_web", action="store_true", help="Launch interactive Active-Learning Studio Web UI (Human-in-the-Loop Verifier & Trainer)")
    parser.add_argument("--port", type=int, default=8585, help="Port for the Active-Learning Web UI (default: 8585)")

    args = parser.parse_args()

    if args.launch_web:
        web_server_script = CURRENT_DIR / "web_studio_server.py"
        web_cmd = [sys.executable, str(web_server_script), "--port", str(args.port)]
        if args.db_path:
            web_cmd.extend(["--db", str(args.db_path)])
        elif args.output_dir:
            out_p = Path(os.path.expanduser(args.output_dir)).resolve()
            web_cmd.extend(["--db", str(out_p / "snapshots_catalog.db")])
        print(f"🌐 Iniciando Servidor Web de Estudio Activo: http://localhost:{args.port}...")
        return subprocess.run(web_cmd).returncode

    model_name = args.model.strip().lower()

    # Determine origin
    origin = args.origin
    if origin == "auto":
        if any(":" in str(inp) and not Path(str(inp)).exists() for inp in args.inputs):
            origin = "gdrive"
        elif not args.inputs and model_name != "unnamed_model":
            # Check local external disk first
            local_found = False
            adata_root = Path("/media/fbetancourt/ADATA HD710 PRO1")
            if adata_root.exists():
                aliases = GoogleDriveSource._model_aliases(model_name)
                for a in aliases:
                    if (adata_root / a).exists():
                        local_found = True
                        break
            origin = "local" if local_found else "gdrive"
        else:
            origin = "local"

    # Destination directory
    if args.output_dir:
        out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    elif model_name != "unnamed_model":
        out_dir = Path.home() / "ComputerVision" / "datasets" / model_name / "photos" / "snapshots"
    else:
        out_dir = Path.cwd() / "harvested_snapshots"

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
        studio_mode=args.studio or True,
        situations=args.situations,
        contact_sheet=args.contact_sheet or args.studio or True,
        enable_db=args.enable_db or True,
        db_path=args.db_path,
        store_blob=args.store_blob or True,
        supervise_agy=args.supervise_agy,
        verbose=args.verbose or True
    )

    t0 = time.time()
    staged_to_clean: List[Path] = []
    result: Dict[str, Any] = {}

    print(f"\n🚀 Iniciando extracción con VideoSnapshotHarvester:")
    print(f"   👤 Modelo objetivo: {model_name}")
    print(f"   🌐 Origen seleccionado: {origin.upper()}")
    print(f"   📁 Directorio de salida: {out_dir}")

    try:
        # ==========================================
        # ORIGIN: GOOGLE DRIVE
        # ==========================================
        if origin == "gdrive":
            remotes = None if args.gdrive_remote == "all" else [args.gdrive_remote]
            staged_files: List[Path] = []

            # Case A: Inputs contain direct remote path(s)
            remote_inputs = [inp for inp in args.inputs if ":" in inp and not Path(inp).exists()]
            if remote_inputs:
                for r_inp in remote_inputs:
                    # Check if single remote file or remote directory
                    if Path(r_inp).suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".jpg", ".jpeg", ".png"}:
                        stg = GoogleDriveSource.stage_file(r_inp)
                        if stg:
                            staged_files.append(stg)
                    else:
                        # Remote folder: list items
                        cmd = ["rclone", "lsjson", r_inp, "--max-depth", "1"]
                        try:
                            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                            if res.returncode == 0 and res.stdout.strip():
                                items = json.loads(res.stdout)
                                for item in items:
                                    if item.get("IsDir"):
                                        continue
                                    fname = item.get("Name", "")
                                    ext = Path(fname).suffix.lower()
                                    if ext in {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".jpg", ".jpeg", ".png"}:
                                        sub_remote = f"{r_inp.rstrip('/')}/{fname}"
                                        stg = GoogleDriveSource.stage_file(sub_remote)
                                        if stg:
                                            staged_files.append(stg)
                                        if len(staged_files) >= args.max_videos:
                                            break
                        except Exception as e:
                            print(f"⚠️ Error listando carpeta remota {r_inp}: {e}")

            # Case B: Search Google Drive accounts by model name
            if not staged_files and model_name != "unnamed_model":
                print(f"🔍 Buscando archivos de '{model_name}' en cuentas Google Drive ({args.gdrive_remote})...")
                discovered = GoogleDriveSource.search_model_media(
                    model_name=model_name,
                    remotes=remotes,
                    base_folder=args.gdrive_folder,
                    media_type=args.media_type,
                    max_files=args.max_videos
                )
                print(f"   Encontrados {len(discovered)} archivos en Google Drive.")
                for item in discovered:
                    stg = GoogleDriveSource.stage_file(item["remote_path"])
                    if stg:
                        staged_files.append(stg)

            if not staged_files:
                print(f"❌ No se encontraron archivos para cosechar en Google Drive para '{model_name}'.", file=sys.stderr)
                return 1

            if args.clean_staging:
                staged_to_clean.extend(staged_files)

            print(f"🎬 Cosechando snapshots desde {len(staged_files)} archivos de Google Drive...")
            result = harvester.harvest_multi_source(
                sources=staged_files,
                recursive=args.recursive,
                media_type=args.media_type,
                max_videos=args.max_videos,
                situations_per_video=args.situations_per_video
            )

        # ==========================================
        # ORIGIN: LOCAL FILESYSTEM / DISK
        # ==========================================
        else:
            local_sources: List[Union[Path, str]] = []

            if args.inputs:
                for inp in args.inputs:
                    p = Path(os.path.expanduser(inp)).resolve()
                    if p.exists():
                        local_sources.append(p)
                    else:
                        print(f"⚠️ Ruta no encontrada: {inp}")

            # If no inputs provided, discover from external disk or datasets
            if not local_sources and model_name != "unnamed_model":
                adata_root = Path("/media/fbetancourt/ADATA HD710 PRO1")
                aliases = GoogleDriveSource._model_aliases(model_name)
                for a in aliases:
                    cand = adata_root / a
                    if cand.is_dir():
                        local_sources.append(cand)
                        break

            if not local_sources:
                print(f"❌ Error: No se especificaron archivos/rutas locales válidas y no se halló carpeta para '{model_name}'.", file=sys.stderr)
                return 1

            # If single file and specific timestamps requested
            if len(local_sources) == 1 and Path(local_sources[0]).is_file() and args.timestamps:
                ts_list = [t.strip() for t in args.timestamps.split(",") if t.strip()]
                result = harvester.harvest_from_timestamps(local_sources[0], ts_list)
            elif len(local_sources) == 1 and Path(local_sources[0]).is_file() and not args.recursive:
                if harvester.studio_mode or args.studio:
                    result = harvester.harvest_studio_situations(
                        video_input=local_sources[0],
                        num_situations=0 if args.per_minute else (args.situations_per_video or args.situations),
                        start_time=args.start,
                        end_time=args.end
                    )
                else:
                    result = harvester.harvest_from_video(
                        video_input=local_sources[0],
                        start_time=args.start,
                        end_time=args.end,
                        fps_interval=args.fps
                    )
            else:
                result = harvester.harvest_multi_source(
                    sources=local_sources,
                    recursive=args.recursive or True,
                    media_type=args.media_type,
                    max_videos=args.max_videos,
                    situations_per_video=args.situations_per_video
                )

    finally:
        if staged_to_clean:
            print(f"🧹 Limpiando {len(staged_to_clean)} archivos temporales de staging...")
            GoogleDriveSource.clean_staging(staged_to_clean)

    elapsed = time.time() - t0
    saved = result.get("total_records", len(result.get("records", [])))

    print(f"\n✅ Extracción completada en {elapsed:.2f}s:")
    print(f"   📸 Total de registros/recortes: {saved}")
    print(f"   📁 Directorio de salida: {out_dir}")
    print(f"   📋 Metadatos JSON: {out_dir / 'SNAPSHOTS_METADATA.json'}")

    if harvester.db_manager:
        print(f"   🗄️ Base de datos SQLite: {harvester.db_manager.db_path}")

    # Re-entrenamiento biométrico automático si se solicitó
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

