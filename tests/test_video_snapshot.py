#!/usr/bin/env python3
"""
Unit tests for Video Snapshot Harvester (`video_snapshot_harvester.py`).
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_snapshot_harvester import (
    SharpnessAssessor,
    PerceptualDeduplicator,
    StudioQualityScorer,
    SkinToneAnalyzer,
    FacialMorphologyAnalyzer,
    AnthropometricEstimator,
    AreolaMorphologyAnalyzer,
    SilhouetteMorphologyAnalyzer,
    SnapshotDatabaseManager,
    VideoMetadataExtractor,
    VideoSnapshotHarvester,
    VideoSnapshotPipeline,
    HarvestConfig,
    format_snapshot_filename
)


class TestNamingAndFilenames(unittest.TestCase):
    def test_filename_with_minutes_and_seconds(self):
        filename = format_snapshot_filename("eva_session_01", 84.500)
        self.assertEqual(filename, "eva_session_01_snap_t01m24s_500ms.jpg")
        self.assertIn("eva_session_01", filename)
        self.assertIn("01m24s_500ms", filename)

    def test_filename_with_hours(self):
        filename = format_snapshot_filename("long_stream", 3672.125)
        self.assertEqual(filename, "long_stream_snap_t01h01m12s_125ms.jpg")
        self.assertIn("long_stream", filename)
        self.assertIn("01h01m12s_125ms", filename)

    def test_filename_with_crop_tag(self):
        filename = format_snapshot_filename("eva_session_01", 45.0, tag="face")
        self.assertEqual(filename, "eva_session_01_snap_t00m45s_000ms_face.jpg")


class TestSharpnessAssessor(unittest.TestCase):
    def test_sharp_image_detection(self):
        sharp_img = np.full((200, 200, 3), 128, dtype=np.uint8)
        sharp_img[::10, :] = 255
        sharp_img[:, ::10] = 0
        variance = SharpnessAssessor.compute_laplacian_variance(sharp_img)
        self.assertGreater(variance, 100.0)
        self.assertTrue(SharpnessAssessor.is_sharp(sharp_img, min_variance=75.0))

    def test_blurry_image_rejection(self):
        blank_img = np.full((200, 200, 3), 128, dtype=np.uint8)
        variance = SharpnessAssessor.compute_laplacian_variance(blank_img)
        self.assertEqual(variance, 0.0)
        self.assertFalse(SharpnessAssessor.is_sharp(blank_img, min_variance=75.0))


class TestStudioQualityScorer(unittest.TestCase):
    def test_ideal_studio_frame(self):
        frame = np.full((200, 200, 3), 130, dtype=np.uint8)
        cv2.circle(frame, (100, 100), 50, (60, 60, 60), -1)
        cv2.circle(frame, (100, 100), 25, (200, 200, 200), -1)
        score_info = StudioQualityScorer.score_frame(frame, min_sharpness=20.0, person_conf=0.9)
        self.assertTrue(score_info["is_studio_worthy"])
        self.assertGreater(score_info["studio_score"], 40.0)

    def test_rejects_dark_or_blown_out_frames(self):
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        score_info = StudioQualityScorer.score_frame(dark_frame)
        self.assertFalse(score_info["is_studio_worthy"])
        self.assertEqual(score_info["reason"], "extreme_lighting")


class TestPerceptualDeduplicator(unittest.TestCase):
    def test_identical_images_have_zero_hamming_distance(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        hash_val = PerceptualDeduplicator.compute_dhash(img)
        dedup = PerceptualDeduplicator(max_hamming_dist=6)
        self.assertFalse(dedup.is_duplicate(hash_val))
        self.assertTrue(dedup.is_duplicate(hash_val))


class TestMorphometricAndAnthropometricAnalyzers(unittest.TestCase):
    """Test suite for forensic skin, facial features, anthropometrics, and areola morphology."""

    def test_skin_tone_analysis(self):
        # Create a patch of warm peach human skin tone (BGR: ~140, 175, 220)
        skin_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        skin_patch[:] = (140, 175, 220)
        skin_info = SkinToneAnalyzer.analyze_skin(skin_patch)
        self.assertTrue(skin_info["hex"].startswith("#"))
        self.assertIn(skin_info["fitzpatrick_type"], ["Type I", "Type II", "Type III", "Type IV"])

    def test_facial_morphology_analysis(self):
        # Create synthetic face crop
        face_img = np.full((140, 100, 3), (140, 175, 220), dtype=np.uint8)
        # Hair at the top
        face_img[:20, :] = (20, 20, 20)
        # Eyes with blue tones
        face_img[45:55, 25:40] = (200, 100, 40)
        face_img[45:55, 60:75] = (200, 100, 40)
        # Nose and mouth
        face_img[65:85, 45:55] = (120, 150, 190)
        face_img[100:110, 35:65] = (60, 60, 180)

        face_meta = FacialMorphologyAnalyzer.analyze_face(face_img)
        self.assertIn("face_shape", face_meta)
        self.assertIn("eye_color", face_meta)
        self.assertIn("hair_color", face_meta)
        self.assertIn("nose_shape", face_meta)
        self.assertIn("ear_shape", face_meta)
        self.assertTrue(face_meta["eye_color_hex"].startswith("#"))

    def test_areola_morphology_analysis(self):
        # Synthetic areola: light background with darker elliptical center
        breast_patch = np.full((120, 120, 3), (160, 190, 230), dtype=np.uint8)
        cv2.ellipse(breast_patch, (60, 60), (30, 26), 0, 0, 360, (70, 95, 140), -1)
        areola_meta = AreolaMorphologyAnalyzer.analyze_areola(breast_patch)
        self.assertIn(areola_meta["shape"], ["Circular (Redonda)", "Ovalada Horizontal", "Ovalada Vertical", "Elíptica Regular"])
        self.assertGreater(areola_meta["aspect_ratio"], 0.8)

    def test_anthropometric_stature_and_hands(self):
        synthetic_frame = np.full((640, 480, 3), 128, dtype=np.uint8)
        rois_data = {
            "boxes": {
                "face": [190, 40, 290, 160],       # Height: 120px
                "torso": [160, 160, 320, 360],
                "hips_glutes": [150, 360, 330, 520],
                "hand": [110, 320, 200, 410]
            }
        }
        anthro = AnthropometricEstimator.estimate(synthetic_frame, rois_data)
        self.assertGreater(anthro["estimated_height_cm"], 140.0)
        self.assertLess(anthro["estimated_height_cm"], 190.0)
        self.assertIn("Proporcionada", anthro["hand_size_category"])


class TestSnapshotDatabaseManager(unittest.TestCase):
    """Test suite for SQLite cataloging, indexed queries, and image BLOB storage."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_db_")
        self.db_path = os.path.join(self.test_dir, "test_catalog.db")
        self.db_manager = SnapshotDatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_upsert_and_query_by_zone(self):
        record_front = {
            "model_name": "eva",
            "image_filename": "eva_snap_t01m24s_frente.jpg",
            "image_path": "/fake/path/eva_snap_t01m24s_frente.jpg",
            "zone_type": "frente",
            "description": "Frontal pelvic crop of eva",
            "source_video": "eva_stream.mp4",
            "source_video_path": "/fake/eva_stream.mp4",
            "timestamp_sec": 84.5,
            "timestamp_str": "01:24.500",
            "studio_score": 82.0,
            "sharpness": 115.0,
            "brightness": 132.0,
            "contrast": 48.0,
            "skin_tone_hex": "#e2b292",
            "fitzpatrick_type": "Type II",
            "silhouette_type": "Hourglass",
            "areola_shape": "N/A",
            "eye_color": "Hazel",
            "hair_color": "Dark Brown",
            "face_shape": "Oval",
            "nose_shape": "Straight",
            "ear_shape": "Proportional",
            "estimated_height_cm": 166.0,
            "hand_size_category": "Proportional",
            "biometrics": {"fitz": "II"}
        }

        # Test insert
        self.db_manager.upsert_snapshot(record_front, image_blob=b"FAKE_JPEG_BYTES")
        self.assertEqual(self.db_manager.count("eva"), 1)

        # Test query by zone
        frente_rows = self.db_manager.query_by_zone("frente", "eva")
        self.assertEqual(len(frente_rows), 1)
        self.assertEqual(frente_rows[0]["zone_type"], "frente")
        self.assertEqual(frente_rows[0]["image_blob"], b"FAKE_JPEG_BYTES")
        self.assertEqual(frente_rows[0]["eye_color"], "Hazel")


class TestEndToEndVideoHarvesting(unittest.TestCase):
    """End-to-end integration test creating a synthetic video and running extraction with SQLite & JSON."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_video_harvest_")
        self.video_path = os.path.join(self.test_dir, "synthetic_model_clip.mp4")
        self.output_dir = os.path.join(self.test_dir, "snapshots_out")

        width, height = 320, 240
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        for i in range(30):
            frame = np.full((height, width, 3), 128, dtype=np.uint8)
            cv2.putText(frame, f"Model Scene {i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(frame, (i * 8, 20), (i * 8 + 50, 80), (30, 200, 30), -1)
            cv2.circle(frame, (250, 160), 30, (220, 50, 50), -1)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_harvest_generates_json_and_sqlite(self):
        db_file = os.path.join(self.output_dir, "catalog.db")
        config = HarvestConfig(
            mode="interval",
            interval_sec=0.3,
            min_sharpness=10.0,
            max_snapshots=4,
            enable_db=True,
            db_path=db_file,
            store_blob=True
        )

        pipeline = VideoSnapshotPipeline(self.video_path, self.output_dir, config)
        manifest = pipeline.harvest()

        self.assertGreater(manifest["total_records"], 0)
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, "SNAPSHOTS_METADATA.json")))
        self.assertTrue(os.path.isfile(db_file))

        with open(os.path.join(self.output_dir, "SNAPSHOTS_METADATA.json"), "r") as f:
            data = json.load(f)
            self.assertEqual(data["source_video"], "synthetic_model_clip.mp4")
            first_rec = data["records"][0]
            self.assertIn("zone_type", first_rec)
            self.assertIn("skin_tone_hex", first_rec)
            self.assertIn("eye_color", first_rec)
            self.assertIn("face_shape", first_rec)
            self.assertIn("estimated_height_cm", first_rec)

        db = SnapshotDatabaseManager(db_file)
        self.assertGreater(db.count(), 0)


class TestAnatomicalZoneExtractor(unittest.TestCase):
    def test_granular_lateral_zones_extraction(self):
        from video_snapshot_harvester import AnatomicalZoneExtractor

        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        # Mock facial crop
        face_crop = np.full((200, 160, 3), 180, dtype=np.uint8)
        # Mock torso crop
        torso_crop = np.full((300, 240, 3), 170, dtype=np.uint8)
        # Mock hips crop
        hips_crop = np.full((250, 260, 3), 160, dtype=np.uint8)

        rois = {
            "face": face_crop,
            "torso": torso_crop,
            "hips_glutes": hips_crop
        }

        # Mock COCO-17 keypoints
        kpts = np.zeros((17, 3), dtype=float)
        # Shoulders (5, 6)
        kpts[5] = [500, 200, 0.95]
        kpts[6] = [700, 200, 0.95]
        # Elbows (7, 8)
        kpts[7] = [460, 320, 0.90]
        kpts[8] = [740, 320, 0.90]
        # Wrists (9, 10)
        kpts[9] = [430, 420, 0.88]
        kpts[10] = [770, 420, 0.88]
        # Hips (11, 12)
        kpts[11] = [530, 430, 0.95]
        kpts[12] = [670, 430, 0.95]
        # Knees (13, 14)
        kpts[13] = [540, 560, 0.90]
        kpts[14] = [660, 560, 0.90]
        # Ankles (15, 16)
        kpts[15] = [545, 680, 0.85]
        kpts[16] = [655, 680, 0.85]

        zones = AnatomicalZoneExtractor.extract_all_zones(frame, rois, kpts=kpts)

        # Facial zones
        self.assertIn("cara", zones)
        self.assertIn("frente", zones)
        self.assertIn("cejas", zones)
        self.assertIn("ojos", zones)
        self.assertIn("nariz", zones)
        self.assertIn("boca", zones)
        self.assertIn("menton", zones)
        self.assertIn("oreja_izq", zones)
        self.assertIn("oreja_der", zones)
        self.assertIn("cabello", zones)

        # Torso & trunk zones
        self.assertIn("torso", zones)
        self.assertIn("breast", zones)
        self.assertIn("aerola_izq", zones)
        self.assertIn("aerola_der", zones)
        self.assertIn("ombligo", zones)
        self.assertIn("cadera", zones)
        self.assertIn("espalda", zones)

        # Pelvic & intimate
        self.assertIn("entrepierna", zones)
        self.assertIn("pompi", zones)

        # Upper limbs (left & right)
        self.assertIn("brazo_izq", zones)
        self.assertIn("brazo_der", zones)
        self.assertIn("mano_izq", zones)
        self.assertIn("mano_der", zones)

        # Lower limbs (left & right)
        self.assertIn("pierna_izq", zones)
        self.assertIn("pierna_der", zones)
        self.assertIn("muslo_izq", zones)
        self.assertIn("muslo_der", zones)
        self.assertIn("tobillo_izq", zones)
        self.assertIn("tobillo_der", zones)
        self.assertIn("pie_izq", zones)
        self.assertIn("pie_der", zones)


class TestGoogleDriveSourceAndOrigin(unittest.TestCase):
    def test_configured_remotes_detection(self):
        from video_snapshot_harvester import GoogleDriveSource
        remotes = GoogleDriveSource.get_configured_remotes()
        self.assertIsInstance(remotes, list)
        self.assertGreater(len(remotes), 0)

    def test_model_alias_resolution(self):
        from video_snapshot_harvester import GoogleDriveSource
        aliases_uw = GoogleDriveSource._model_aliases("uwxo")
        self.assertIn("uwuxo", aliases_uw)

        aliases_tanni = GoogleDriveSource._model_aliases("tanni_bc")
        self.assertIn("taanni_bc", aliases_tanni)

        aliases_sunshine = GoogleDriveSource._model_aliases("sunshine13_")
        self.assertIn("sunshine13__", aliases_sunshine)

    def test_clean_staging_purges_temporary_files(self):
        from video_snapshot_harvester import GoogleDriveSource
        tmp_dir = tempfile.mkdtemp()
        dummy_file = Path(tmp_dir) / "dummy_staged.mp4"
        dummy_file.write_text("sample content")
        self.assertTrue(dummy_file.is_file())

        GoogleDriveSource.clean_staging([dummy_file])
        self.assertFalse(dummy_file.is_file())
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
