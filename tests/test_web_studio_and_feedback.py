#!/usr/bin/env python3
"""
test_web_studio_and_feedback.py - Unit and Integration Tests for Antigravity Active Studio Web Server,
Human-in-the-Loop Verification, Cognitive Feedback Memory, and Omni-Source APIs.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_snapshot_harvester import SnapshotDatabaseManager
from web_studio_server import create_app
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop


class TestSnapshotDatabaseFeedbackAndStatus(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test_feedback.db"
        self.db = SnapshotDatabaseManager(self.db_path)

        # Seed sample record
        self.sample_rec = {
            "model_name": "sunshine13_",
            "image_filename": "sunshine_01.jpg",
            "image_path": str(Path(self.tmp_dir) / "sunshine_01.jpg"),
            "zone_type": "aerola_izq",
            "description": "Left areola test crop",
            "source_video": "sunshine_stream.mp4",
            "source_video_path": "/fake/sunshine_stream.mp4",
            "timestamp_sec": 42.5,
            "timestamp_str": "00m42s_500ms",
            "studio_score": 88.5,
            "sharpness": 120.0,
            "brightness": 115.0,
            "contrast": 55.0,
            "skin_tone_hex": "#e4b59b",
            "fitzpatrick_type": "Type II",
            "complexion_smoothness": 85.0,
            "silhouette_type": "Hourglass",
            "areola_shape": "Circular",
            "eye_color": "Green",
            "hair_color": "Blonde",
            "face_shape": "Heart",
            "nose_shape": "Straight",
            "ear_shape": "Proportional",
            "estimated_height_cm": 168.0,
            "hand_size_category": "Proportional",
            "status": "pending"
        }
        # Fake image bytes
        self.fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xd9"
        self.snap_id = self.db.upsert_snapshot(self.sample_rec, image_blob=self.fake_jpeg)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_set_status_verified(self):
        ok = self.db.set_status(self.snap_id, status="verified")
        self.assertTrue(ok)
        snap = self.db.get_snapshot_by_id(self.snap_id)
        self.assertEqual(snap["status"], "verified")

    def test_set_status_rejected_with_note(self):
        ok = self.db.set_status(self.snap_id, status="rejected", note="Borroso por movimiento")
        self.assertTrue(ok)
        snap = self.db.get_snapshot_by_id(self.snap_id)
        self.assertEqual(snap["status"], "rejected")
        self.assertEqual(snap["review_note"], "Borroso por movimiento")

    def test_record_feedback_memory_for_agy(self):
        f_id = self.db.record_feedback(
            snapshot_id=self.snap_id,
            model_name="sunshine13_",
            zone_type="aerola_izq",
            error_category="Corte desplazado",
            user_note="El recorte tomó tela del top en vez del pezón"
        )
        self.assertGreater(f_id, 0)

        # Check in memory query
        memories = self.db.get_feedback_memory(model_name="sunshine13_")
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["error_category"], "Corte desplazado")
        self.assertIn("tela del top", memories[0]["user_note"])

        # Check that snapshot was marked as rejected
        snap = self.db.get_snapshot_by_id(self.snap_id)
        self.assertEqual(snap["status"], "rejected")

    def test_get_models_summary(self):
        summary = self.db.get_models_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["model_name"], "sunshine13_")
        self.assertEqual(summary[0]["total"], 1)

    def test_get_image_blob(self):
        data = self.db.get_image_blob(self.snap_id)
        self.assertIsNotNone(data)
        blob, fname = data
        self.assertEqual(blob, self.fake_jpeg)
        self.assertEqual(fname, "sunshine_01.jpg")


class TestWebStudioAPI(AioHTTPTestCase):
    async def get_application(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "web_test.db"
        self.db = SnapshotDatabaseManager(self.db_path)

        self.fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xd9"
        rec = {
            "model_name": "taanni_bc",
            "image_filename": "taanni_crop01.jpg",
            "image_path": "/fake/taanni_crop01.jpg",
            "zone_type": "cara",
            "description": "Face crop",
            "source_video": "taanni.mp4",
            "source_video_path": "/fake/taanni.mp4",
            "timestamp_sec": 10.0,
            "timestamp_str": "00m10s_000ms",
            "studio_score": 92.0,
            "status": "pending"
        }
        self.snap_id = self.db.upsert_snapshot(rec, image_blob=self.fake_jpeg)
        return create_app(self.db_path)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @unittest_run_loop
    async def test_dashboard_renders(self):
        resp = await self.client.request("GET", "/")
        self.assertEqual(resp.status, 200)
        text = await resp.text()
        self.assertIn("Antigravity Active Studio", text)

    @unittest_run_loop
    async def test_api_models(self):
        resp = await self.client.request("GET", "/api/models")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 1)
        self.assertEqual(data["models"][0]["model_name"], "taanni_bc")

    @unittest_run_loop
    async def test_api_snapshots_list(self):
        resp = await self.client.request("GET", "/api/snapshots?model=taanni_bc")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["zone_type"], "cara")

    @unittest_run_loop
    async def test_api_image_streaming(self):
        resp = await self.client.request("GET", f"/api/image/{self.snap_id}")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.content_type, "image/jpeg")
        body = await resp.read()
        self.assertEqual(body, self.fake_jpeg)

    @unittest_run_loop
    async def test_api_verify(self):
        resp = await self.client.request("POST", f"/api/verify/{self.snap_id}")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "verified")

    @unittest_run_loop
    async def test_api_reject_with_note(self):
        payload = {"note": "Ojos cerrados", "error_category": "Calidad Facial"}
        resp = await self.client.request("POST", f"/api/reject/{self.snap_id}", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "rejected")

        # Verify feedback memory received it
        mem_resp = await self.client.request("GET", "/api/agy/memory")
        mem_data = await mem_resp.json()
        self.assertEqual(mem_data["count"], 1)
        self.assertEqual(mem_data["memories"][0]["error_category"], "Calidad Facial")

    @unittest_run_loop
    async def test_api_videos_and_scenes_and_search(self):
        # Insert a sample scene
        scene_id = self.db.upsert_scene(
            model_name="taanni_bc",
            video_filename="taanni.mp4",
            video_path="/fake/taanni.mp4",
            minute_index=2,
            timestamp_sec=120.0,
            timestamp_str="02m00s",
            scene_description="Escena minuto 2. Plano General. Tono de piel Tan.",
            tags="taanni_bc, minuto_2, plano general",
            studio_score=85.0,
            image_blob=self.fake_jpeg
        )
        self.assertGreater(scene_id, 0)

        # Test /api/videos
        v_resp = await self.client.request("GET", "/api/videos?model=taanni_bc")
        self.assertEqual(v_resp.status, 200)
        v_data = await v_resp.json()
        self.assertEqual(v_data["count"], 1)
        self.assertEqual(v_data["videos"][0]["filename"], "taanni.mp4")

        # Test /api/video/scenes
        s_resp = await self.client.request("GET", "/api/video/scenes?video=taanni.mp4")
        self.assertEqual(s_resp.status, 200)
        s_data = await s_resp.json()
        self.assertEqual(s_data["count"], 1)
        self.assertEqual(s_data["scenes"][0]["minute_index"], 2)

        # Test /api/video/scene_image/{id}
        img_resp = await self.client.request("GET", f"/api/video/scene_image/{scene_id}")
        self.assertEqual(img_resp.status, 200)
        img_body = await img_resp.read()
        self.assertEqual(img_body, self.fake_jpeg)

        # Test /api/search by minute
        sr1 = await self.client.request("GET", "/api/search?q=minuto%202&model=taanni_bc")
        self.assertEqual(sr1.status, 200)
        srd1 = await sr1.json()
        self.assertEqual(srd1["count"], 1)
        self.assertEqual(srd1["results"][0]["minute_index"], 2)

        # Test /api/search by FTS text
        sr2 = await self.client.request("GET", "/api/search?q=Plano%20General&model=taanni_bc")
        self.assertEqual(sr2.status, 200)
        srd2 = await sr2.json()
        self.assertEqual(srd2["count"], 1)


if __name__ == "__main__":
    unittest.main()

