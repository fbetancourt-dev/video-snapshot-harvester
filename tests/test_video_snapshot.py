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
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from video_snapshot_harvester import (
    SharpnessAssessor,
    PerceptualDeduplicator,
    StudioQualityScorer,
    VideoMetadataExtractor,
    VideoSnapshotHarvester,
    VideoSnapshotPipeline,
    HarvestConfig,
    format_snapshot_filename
)


class TestNamingAndFilenames(unittest.TestCase):
    """Test suite ensuring snapshots contain video name and exact extracted timestamp."""

    def test_filename_with_minutes_and_seconds(self):
        filename = format_snapshot_filename("eva_session_01", 84.500)
        # 84.5s = 1m 24s 500ms
        self.assertEqual(filename, "eva_session_01_snap_t01m24s_500ms.jpg")
        self.assertIn("eva_session_01", filename)
        self.assertIn("01m24s_500ms", filename)

    def test_filename_with_hours(self):
        filename = format_snapshot_filename("long_stream", 3672.125)
        # 3672.125s = 1h 01m 12s 125ms
        self.assertEqual(filename, "long_stream_snap_t01h01m12s_125ms.jpg")
        self.assertIn("long_stream", filename)
        self.assertIn("01h01m12s_125ms", filename)

    def test_filename_with_crop_tag(self):
        filename = format_snapshot_filename("eva_session_01", 45.0, tag="face")
        self.assertEqual(filename, "eva_session_01_snap_t00m45s_000ms_face.jpg")


class TestSharpnessAssessor(unittest.TestCase):
    """Test suite for Laplacian variance sharpness filtering."""

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
    """Test suite for studio aesthetic quality scoring."""

    def test_ideal_studio_frame(self):
        # Create well-lit frame with contrast and sharp edges (ideal studio conditions)
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
    """Test suite for 64-bit difference hash (dHash) deduplication."""

    def test_identical_images_have_zero_hamming_distance(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        hash_val = PerceptualDeduplicator.compute_dhash(img)
        
        dedup = PerceptualDeduplicator(max_hamming_dist=6)
        self.assertFalse(dedup.is_duplicate(hash_val))
        self.assertTrue(dedup.is_duplicate(hash_val))

    def test_distinguishable_images_are_kept(self):
        dedup = PerceptualDeduplicator(max_hamming_dist=6)
        
        img1 = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        img2 = np.tile(np.linspace(0, 255, 100, dtype=np.uint8).reshape(-1, 1), (1, 100))
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        h1 = PerceptualDeduplicator.compute_dhash(img1)
        h2 = PerceptualDeduplicator.compute_dhash(img2)
        
        self.assertFalse(dedup.is_duplicate(h1))
        self.assertFalse(dedup.is_duplicate(h2))


class TestVideoHarvestingPipeline(unittest.TestCase):
    """End-to-end integration test creating a synthetic video and running extraction."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_video_harvest_")
        self.video_path = os.path.join(self.test_dir, "synthetic_model_clip.mp4")
        self.output_dir = os.path.join(self.test_dir, "snapshots_out")

        # Generate a small 1-second synthetic 30fps video with proper studio lighting background
        width, height = 320, 240
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))
        
        for i in range(30):
            # Midtone studio background (gray 128)
            frame = np.full((height, width, 3), 128, dtype=np.uint8)
            # Dynamic sharp shapes
            cv2.putText(frame, f"Model Scene {i}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(frame, (i * 8, 20), (i * 8 + 50, 80), (30, 200, 30), -1)
            cv2.circle(frame, (250, 160), 30, (220, 50, 50), -1)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_probe_metadata(self):
        meta = VideoMetadataExtractor.probe(self.video_path)
        self.assertGreater(meta.duration_sec, 0.5)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertAlmostEqual(meta.fps, 30.0, delta=1.0)

    def test_harvest_interval_mode_filenames_and_manifest(self):
        config = HarvestConfig(
            mode="interval",
            interval_sec=0.3,
            min_sharpness=10.0,
            max_snapshots=5,
            generate_thumbnails=True
        )
        
        pipeline = VideoSnapshotPipeline(self.video_path, self.output_dir, config)
        manifest = pipeline.harvest()
        
        self.assertGreater(manifest["total_saved"], 0)
        self.assertTrue(os.path.isdir(self.output_dir))
        
        manifest_file = os.path.join(self.output_dir, "SNAPSHOTS_MANIFEST.json")
        self.assertTrue(os.path.isfile(manifest_file))
        
        with open(manifest_file, "r") as f:
            data = json.load(f)
            self.assertEqual(data["video_filename"], "synthetic_model_clip.mp4")
            self.assertGreater(len(data["snapshots"]), 0)
            
            # Verify snapshot filename format contains source video stem and timestamp
            first_snap = data["snapshots"][0]
            self.assertTrue(first_snap["filename"].startswith("synthetic_model_clip_snap_t"))
            self.assertIn("s_", first_snap["filename"])
            self.assertIn("ms.jpg", first_snap["filename"])

    def test_harvest_from_explicit_timestamps(self):
        harvester = VideoSnapshotHarvester(
            output_dir=Path(self.output_dir),
            min_sharpness=10.0
        )
        timestamps = ["00:00:00.200", "00:00:00.600"]
        manifest = harvester.harvest_from_timestamps(self.video_path, timestamps)
        
        self.assertEqual(manifest["total_saved"], 2)
        filenames = [s["filename"] for s in manifest["snapshots"]]
        for fn in filenames:
            self.assertTrue(fn.startswith("synthetic_model_clip_snap_t"))


if __name__ == "__main__":
    unittest.main()
