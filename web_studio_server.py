#!/usr/bin/env python3
"""
web_studio_server.py - Antigravity Continuous Active-Learning Studio & Human-in-the-Loop Web UI.

Features:
1. Pure Python aiohttp async server streaming images directly from SQLite BLOBs or files.
2. Model Selector & 32-Zone Anatomical Filter.
3. Human-in-the-Loop Review: 'Verificado' (✅) vs 'Descartado' (❌) with optional rejection reason.
4. Cognitive Active Learning Feedback Memory for Antigravity (AGY): Stores human correction
   notes and error categories to refine future harvesting.
5. Video Player with HTTP 206 Partial Content (Range requests) seeking support.
6. Minute-by-Minute Visual Timeline: Snapshots of each minute clickable to jump directly in video.
7. Contextual Natural Language Search: Local FTS5 full-text & minute parser with scene descriptions.
8. Omni-Source Ingestion: Trigger on-demand harvesting from Google Drive (multi-account),
   Social Media Reels (Instagram, TikTok, X, Reddit via yt-dlp), Local Folders, or Catalogs.
9. Continuous Model Training: Trigger biometric profile retraining directly from verified datasets.
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from aiohttp import web

# Add local path to import VideoSnapshotHarvester and helpers
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from video_snapshot_harvester import (
    SnapshotDatabaseManager,
    GoogleDriveSource,
    SocialMediaSource,
    CatalogSource,
    ANATOMICAL_ZONES
)

DEFAULT_DB_PATH = Path("/home/fbetancourt/Gemini/training_datasets/master_snapshots.db")
MM_SCRAPER_PATH = Path("/home/fbetancourt/Gemini/model-media-scraper")

# In-memory background harvest jobs tracker
ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}


def find_available_databases(custom_db: Optional[Path] = None) -> List[Path]:
    """Finds all candidate SQLite catalog databases across datasets and projects."""
    dbs = []
    if custom_db and Path(custom_db).is_file():
        dbs.append(Path(custom_db).resolve())

    # Check default locations
    candidates = [
        DEFAULT_DB_PATH,
        CURRENT_DIR / "snapshots_catalog.db",
        Path.home() / "ComputerVision" / "datasets"
    ]
    for c in candidates:
        if c.is_file() and c not in dbs:
            dbs.append(c)
        elif c.is_dir():
            for db_f in c.glob("**/snapshots_catalog.db"):
                if db_f not in dbs:
                    dbs.append(db_f)

    # Check training_datasets folder
    td_root = Path("/home/fbetancourt/Gemini/training_datasets")
    if td_root.exists():
        for db_f in td_root.glob("**/snapshots_catalog.db"):
            if db_f not in dbs:
                dbs.append(db_f)

    if not dbs:
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        dbs.append(DEFAULT_DB_PATH)

    return dbs


class WebStudioApp:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_manager = SnapshotDatabaseManager(self.db_path)
        self.blob_semaphore = asyncio.Semaphore(8)

    async def get_models(self, request: web.Request) -> web.Response:
        models = await asyncio.to_thread(self.db_manager.get_models_summary)
        return web.json_response({"models": models, "db_path": str(self.db_path)})

    async def get_snapshots(self, request: web.Request) -> web.Response:
        model = request.query.get("model", None)
        zone = request.query.get("zone", None)
        status = request.query.get("status", "all")
        page = max(1, int(request.query.get("page", 1)))
        page_size = max(1, min(100, int(request.query.get("page_size", 24))))
        offset = (page - 1) * page_size

        if zone == "all":
            zone = None
        if model == "all":
            model = None

        items, total = await asyncio.to_thread(
            self.db_manager.get_snapshots_filtered,
            model_name=model,
            zone_type=zone,
            status=status,
            limit=page_size,
            offset=offset
        )

        total_pages = max(1, (total + page_size - 1) // page_size)
        return web.json_response({
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        })

    async def get_image(self, request: web.Request) -> web.Response:
        snap_id = int(request.match_info["id"])
        async with self.blob_semaphore:
            data = await asyncio.to_thread(self.db_manager.get_image_blob, snap_id)
        if not data:
            return web.Response(status=404, text="Image not found")
        blob, fname = data
        return web.Response(
            body=blob,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Content-Disposition": f'inline; filename="{fname}"'
            }
        )

    async def verify_snapshot(self, request: web.Request) -> web.Response:
        snap_id = int(request.match_info["id"])
        success = self.db_manager.set_status(snap_id, status="verified")
        return web.json_response({"success": success, "id": snap_id, "status": "verified"})

    async def reject_snapshot(self, request: web.Request) -> web.Response:
        snap_id = int(request.match_info["id"])
        try:
            body = await request.json()
        except Exception:
            body = {}
        note = body.get("note", "").strip()
        error_category = body.get("error_category", "Calidad / Descarte").strip()

        # If note or error category provided, record in cognitive feedback memory for AGY
        snap = self.db_manager.get_snapshot_by_id(snap_id)
        if snap and (note or error_category != "Calidad / Descarte"):
            self.db_manager.record_feedback(
                snapshot_id=snap_id,
                model_name=snap.get("model_name", "unknown"),
                zone_type=snap.get("zone_type", "cuerpo"),
                error_category=error_category,
                user_note=note
            )
        else:
            self.db_manager.set_status(snap_id, status="rejected", note=note if note else None)

        return web.json_response({"success": True, "id": snap_id, "status": "rejected", "note": note})

    async def record_feedback(self, request: web.Request) -> web.Response:
        snap_id = int(request.match_info["id"])
        body = await request.json()
        error_category = body.get("error_category", "Corte / Detección Incorrecta")
        user_note = body.get("user_note", "").strip()

        snap = self.db_manager.get_snapshot_by_id(snap_id)
        if not snap:
            return web.json_response({"success": False, "error": "Snapshot not found"}, status=404)

        f_id = self.db_manager.record_feedback(
            snapshot_id=snap_id,
            model_name=snap.get("model_name", "unknown"),
            zone_type=snap.get("zone_type", "cuerpo"),
            error_category=error_category,
            user_note=user_note
        )
        return web.json_response({"success": True, "feedback_id": f_id, "snapshot_id": snap_id})

    async def get_agy_memory(self, request: web.Request) -> web.Response:
        model = request.query.get("model", None)
        zone = request.query.get("zone", None)
        memories = self.db_manager.get_feedback_memory(model_name=model, zone_type=zone)
        return web.json_response({"memories": memories, "count": len(memories)})

    async def get_master_snapshots(self, request: web.Request) -> web.Response:
        model = request.query.get("model", "all")
        page = max(1, int(request.query.get("page", 1)))
        limit = max(1, min(100, int(request.query.get("limit", 24))))
        offset = (page - 1) * limit

        def _fetch_masters():
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                base_cond = "(is_master_snapshot=1 OR zone_type='cuerpo' OR zone_type='master_frame')"
                params = []
                if model != "all":
                    base_cond += " AND model_name=?"
                    params.append(model)

                c.execute(f"SELECT COUNT(*) FROM snapshots WHERE {base_cond}", tuple(params))
                total = c.fetchone()[0]

                sql = f"""
                    SELECT s.id, s.model_name, s.image_filename, s.source_video, s.timestamp_str,
                           s.studio_score, s.sharpness, s.status, s.created_at,
                           (SELECT COUNT(*) FROM snapshots c WHERE c.parent_snapshot_id = s.id) as crops_count
                    FROM snapshots s
                    WHERE {base_cond}
                    ORDER BY s.id DESC
                    LIMIT ? OFFSET ?
                """
                c.execute(sql, tuple(params + [limit, offset]))
                rows = [dict(r) for r in c.fetchall()]
                return rows, total

        masters, total = await asyncio.to_thread(_fetch_masters)
        return web.json_response({
            "masters": masters,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit if total > 0 else 1
        })

    async def get_master_crops(self, request: web.Request) -> web.Response:
        master_id = int(request.match_info["id"])

        def _fetch_crops():
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT id, model_name, image_filename, timestamp_str, source_video, sharpness, studio_score, description FROM snapshots WHERE id=?", (master_id,))
                m_row = c.fetchone()
                master_info = dict(m_row) if m_row else None

                # Child crops: matched by parent_snapshot_id OR same video & near timestamp
                c.execute("""
                    SELECT id, model_name, image_filename, zone_type, studio_score, sharpness, status, review_note, description
                    FROM snapshots
                    WHERE parent_snapshot_id=? OR (parent_snapshot_id IS NULL AND source_video=? AND ABS(timestamp_sec - ?) < 0.2 AND id != ?)
                    ORDER BY id ASC
                """, (master_id, master_info.get("source_video") if master_info else "", 0.0, master_id))
                crops = [dict(r) for r in c.fetchall()]
                return master_info, crops

        master_info, crops = await asyncio.to_thread(_fetch_crops)
        if not master_info:
            return web.json_response({"success": False, "error": "Master snapshot not found"}, status=404)
        return web.json_response({"success": True, "master": master_info, "crops": crops, "crops_count": len(crops)})

    # ==========================================
    # Video & Contextual Scene Search Endpoints
    # ==========================================
    async def get_videos(self, request: web.Request) -> web.Response:
        model = request.query.get("model", None)
        if model == "all":
            model = None

        def _fetch_videos():
            videos_map = {}
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()

                # 1. From video_scenes
                sql_s = "SELECT model_name, video_filename, video_path, COUNT(*) as scene_count FROM video_scenes"
                params_s = []
                if model:
                    sql_s += " WHERE model_name=?"
                    params_s.append(model)
                sql_s += " GROUP BY video_filename"
                c.execute(sql_s, tuple(params_s))
                for row in c.fetchall():
                    fname = row["video_filename"]
                    videos_map[fname] = {
                        "filename": fname,
                        "path": row["video_path"],
                        "model_name": row["model_name"],
                        "scenes_count": row["scene_count"],
                        "exists": Path(row["video_path"]).is_file() if row["video_path"] else False
                    }

                # 2. From snapshots table
                sql_snap = "SELECT model_name, source_video, source_video_path, COUNT(*) as snap_count FROM snapshots WHERE source_video IS NOT NULL"
                params_snap = []
                if model:
                    sql_snap += " AND model_name=?"
                    params_snap.append(model)
                sql_snap += " GROUP BY source_video"
                c.execute(sql_snap, tuple(params_snap))
                for row in c.fetchall():
                    fname = row["source_video"]
                    if fname not in videos_map:
                        videos_map[fname] = {
                            "filename": fname,
                            "path": row["source_video_path"],
                            "model_name": row["model_name"],
                            "scenes_count": 0,
                            "exists": Path(row["source_video_path"]).is_file() if row["source_video_path"] else False
                        }
                    videos_map[fname]["snapshots_count"] = row["snap_count"]

            return list(videos_map.values())

        video_list = await asyncio.to_thread(_fetch_videos)
        return web.json_response({"videos": video_list, "count": len(video_list)})

    async def stream_video(self, request: web.Request) -> web.StreamResponse:
        video_filename = request.query.get("filename", "")
        raw_path = request.query.get("path", "")

        resolved_path = None
        if raw_path and Path(raw_path).is_file():
            resolved_path = Path(raw_path)
        elif video_filename:
            # Query db for path
            with sqlite3.connect(str(self.db_path)) as conn:
                c = conn.cursor()
                c.execute("SELECT video_path FROM video_scenes WHERE video_filename=? LIMIT 1", (video_filename,))
                row = c.fetchone()
                if row and row[0] and Path(row[0]).is_file():
                    resolved_path = Path(row[0])
                else:
                    c.execute("SELECT source_video_path FROM snapshots WHERE source_video=? LIMIT 1", (video_filename,))
                    row2 = c.fetchone()
                    if row2 and row2[0] and Path(row2[0]).is_file():
                        resolved_path = Path(row2[0])

        if not resolved_path or not resolved_path.exists():
            return web.Response(status=404, text=f"Video file not found or inaccessible: {video_filename}")

        # aiohttp web.FileResponse natively implements HTTP 206 Partial Content (Range headers)
        return web.FileResponse(
            resolved_path,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Type": "video/mp4"
            }
        )

    async def get_scenes(self, request: web.Request) -> web.Response:
        video_filename = request.query.get("video", None)
        model = request.query.get("model", None)
        if model == "all":
            model = None

        scenes = await asyncio.to_thread(
            self.db_manager.get_video_scenes,
            video_filename=video_filename,
            model_name=model
        )
        return web.json_response({"scenes": scenes, "count": len(scenes)})

    async def get_scene_image(self, request: web.Request) -> web.Response:
        scene_id = int(request.match_info["id"])
        async with self.blob_semaphore:
            blob = await asyncio.to_thread(self.db_manager.get_scene_image_blob, scene_id)
        if not blob:
            return web.Response(status=404, text="Scene image not found")
        return web.Response(
            body=blob,
            content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )

    async def search_scenes(self, request: web.Request) -> web.Response:
        query = request.query.get("q", "").strip()
        model = request.query.get("model", None)
        limit = max(1, min(100, int(request.query.get("limit", 30))))

        if not query:
            return web.json_response({"results": [], "query": "", "count": 0})

        results = await asyncio.to_thread(
            self.db_manager.search_scenes,
            query=query,
            model_name=model,
            limit=limit
        )
        return web.json_response({
            "results": results,
            "query": query,
            "model": model,
            "count": len(results)
        })

    # ==========================================
    # Harvest & Training Endpoints
    # ==========================================
    async def trigger_harvest(self, request: web.Request) -> web.Response:
        body = await request.json()
        model_name = body.get("model", "").strip()
        origin = body.get("origin", "auto")
        source_input = body.get("source_input", "").strip()
        max_videos = int(body.get("max_videos", 3))
        situations = int(body.get("situations", 4))
        media_type = body.get("media_type", "all")
        clean_staging = bool(body.get("clean_staging", False))

        job_id = f"job_{int(time.time())}_{os.getpid()}"
        ACTIVE_JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "model": model_name,
            "origin": origin,
            "created_at": time.time(),
            "log": []
        }

        # Run harvest in background thread/task
        asyncio.create_task(self._run_harvest_task(
            job_id=job_id,
            model_name=model_name,
            origin=origin,
            source_input=source_input,
            max_videos=max_videos,
            situations=situations,
            media_type=media_type,
            clean_staging=clean_staging
        ))

        return web.json_response({"success": True, "job_id": job_id})

    async def _run_harvest_task(
        self,
        job_id: str,
        model_name: str,
        origin: str,
        source_input: str,
        max_videos: int,
        situations: int,
        media_type: str,
        clean_staging: bool
    ):
        job = ACTIVE_JOBS.get(job_id)
        cmd = [
            sys.executable,
            str(CURRENT_DIR / "video_snapshot_harvester.py"),
            "-m", model_name,
            "--origin", origin,
            "--db", str(self.db_path),
            "--store-blob",
            "--crop-regions",
            "--studio",
            "--situations-per-video", str(situations),
            "--max-videos", str(max_videos),
            "--media-type", media_type
        ]
        if source_input:
            cmd.append(source_input)
        if clean_staging:
            cmd.append("--clean-staging")

        job["log"].append(f"Iniciando comando: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                job["log"].append(decoded)

        await proc.wait()
        job["status"] = "completed" if proc.returncode == 0 else "failed"
        job["returncode"] = proc.returncode

    async def get_harvest_status(self, request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        job = ACTIVE_JOBS.get(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        return web.json_response(job)

    async def trigger_training(self, request: web.Request) -> web.Response:
        body = await request.json()
        model_name = body.get("model", "").strip()
        if not model_name:
            return web.json_response({"error": "model name required"}, status=400)

        # Count verified samples
        with sqlite3.connect(str(self.db_path)) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM snapshots WHERE model_name=? AND status='verified'", (model_name,))
            verified_count = int(c.fetchone()[0])

        # Run trainer script
        cmd = [
            sys.executable,
            str(MM_SCRAPER_PATH / "model_media_scraper.py"),
            model_name,
            "--train-model"
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return web.json_response({
            "success": proc.returncode == 0,
            "model": model_name,
            "verified_samples_used": verified_count,
            "returncode": proc.returncode,
            "output": stdout.decode("utf-8", errors="replace")[-500:]
        })

    async def render_dashboard(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML_TEMPLATE, content_type="text/html")


# ==============================================================================
# Modern Responsive HTML5 / Tailwind-inspired CSS / Vanilla JS Web UI
# ==============================================================================
DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Antigravity Active Studio - Vision & Timeline Studio</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --success-hover: #059669;
      --danger: #ef4444;
      --danger-hover: #dc2626;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --accent: #8b5cf6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background-color: #0f172a;
      border-bottom: 1px solid #1e293b;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-weight: 700;
      font-size: 1.25rem;
      color: #60a5fa;
    }
    .tabs-nav {
      display: flex;
      gap: 0.5rem;
    }
    .tab-btn {
      background: #1e293b;
      color: var(--text-muted);
      border: 1px solid #334155;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.875rem;
      font-weight: 500;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .tab-btn:hover { color: var(--text); background: #334155; }
    .tab-btn.active {
      background: var(--primary);
      color: white;
      border-color: var(--primary-hover);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .btn {
      background-color: var(--card-border);
      color: var(--text);
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .btn:hover { background-color: #374151; }
    .btn-primary { background-color: var(--primary); color: white; }
    .btn-primary:hover { background-color: var(--primary-hover); }
    .btn-success { background-color: var(--success); color: white; }
    .btn-success:hover { background-color: var(--success-hover); }
    .btn-danger { background-color: var(--danger); color: white; }
    .btn-danger:hover { background-color: var(--danger-hover); }

    .main-container {
      flex: 1;
      padding: 1.5rem;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* Filter Controls Bar */
    .filter-bar {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1.5rem;
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 600;
    }
    select, input[type="text"], input[type="number"] {
      background-color: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 0.45rem 0.75rem;
      border-radius: 6px;
      font-size: 0.875rem;
      outline: none;
    }
    select:focus, input:focus { border-color: var(--primary); }

    /* Snapshots Grid */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1.25rem;
    }
    .card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: #374151;
    }
    .card.verified { border: 2px solid var(--success); }
    .card.rejected { opacity: 0.45; border: 1px dashed var(--danger); filter: grayscale(80%); }

    .card-img-wrapper {
      position: relative;
      width: 100%;
      height: 240px;
      background-color: #0b0f19;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .card-img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      loading: lazy;
    }
    .zone-badge {
      position: absolute;
      top: 0.5rem;
      left: 0.5rem;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(4px);
      color: #93c5fd;
      border: 1px solid #1e3a8a;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .score-badge {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(4px);
      color: #34d399;
      border: 1px solid #065f46;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 600;
    }
    .card-body {
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      flex: 1;
    }
    .card-title {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text);
      display: flex;
      justify-content: space-between;
    }
    .card-desc {
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.3;
      max-height: 2.6rem;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .card-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: auto;
      padding-top: 0.5rem;
      border-top: 1px solid #1e293b;
    }
    .card-actions .btn { flex: 1; justify-content: center; padding: 0.4rem; }

    /* Video Player & Timeline Section */
    .video-view-container {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .player-box {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    #video-player {
      width: 100%;
      max-width: 960px;
      max-height: 540px;
      border-radius: 8px;
      background: #000;
      outline: none;
    }
    .timeline-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 1.25rem;
    }
    .timeline-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    .timeline-scroll {
      display: flex;
      gap: 1rem;
      overflow-x: auto;
      padding-bottom: 0.75rem;
      scroll-behavior: smooth;
    }
    .timeline-scroll::-webkit-scrollbar { height: 8px; }
    .timeline-scroll::-webkit-scrollbar-track { background: #0b0f19; border-radius: 4px; }
    .timeline-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    .timeline-card {
      min-width: 200px;
      max-width: 200px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
    }
    .timeline-card:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
    }
    .timeline-card.active {
      border: 2px solid #60a5fa;
      background: #1e3a8a;
    }
    .timeline-thumb {
      width: 100%;
      height: 110px;
      object-fit: cover;
      background: #000;
    }
    .timeline-body {
      padding: 0.6rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .timeline-time {
      font-size: 0.8rem;
      font-weight: 700;
      color: #38bdf8;
      display: flex;
      justify-content: space-between;
    }
    .timeline-desc {
      font-size: 0.7rem;
      color: var(--text-muted);
      line-height: 1.2;
      max-height: 2.4rem;
      overflow: hidden;
    }

    /* Modal dialog */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }
    .modal-overlay.active { display: flex; }
    .modal-card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
      width: 90%;
      max-width: 500px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .reason-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .chip {
      background: #0f172a;
      border: 1px solid #334155;
      padding: 0.3rem 0.6rem;
      border-radius: 20px;
      font-size: 0.75rem;
      color: var(--text-muted);
      cursor: pointer;
    }
    .chip.active, .chip:hover {
      background: #3b82f6;
      color: white;
      border-color: #60a5fa;
    }

    /* Pagination */
    .pagination {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 1rem;
      margin-top: 2rem;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>🌌 Antigravity Studio</span>
    </div>

    <div class="tabs-nav">
      <button class="tab-btn active" data-tab="tab-masters">🖼️ Snapshots Maestros & Recortes</button>
      <button class="tab-btn" data-tab="tab-atlas">✂️ Triage por Zonas (32 Zonas)</button>
      <button class="tab-btn" data-tab="tab-video">🎬 Videos & Timeline</button>
      <button class="tab-btn" data-tab="tab-search">🔍 Buscador Contextual</button>
      <button class="tab-btn" data-tab="tab-memory">🧠 Memoria AGY</button>
      <button class="tab-btn" data-tab="tab-harvest">🚀 Cosecha & Entrenar</button>
    </div>

    <div class="header-actions">
      <button id="btn-refresh" class="btn">🔄 Refrescar</button>
    </div>
  </header>

  <div class="main-container">

    <!-- TAB 0: SNAPSHOTS MAESTROS ORIGINALES Y RECORTES JERÁRQUICOS -->
    <div id="tab-masters" class="tab-pane active">
      <div class="filter-bar">
        <div class="form-group">
          <label for="master-model-select">Modelo Biométrico</label>
          <select id="master-model-select">
            <option value="all">Todos los Modelos</option>
          </select>
        </div>
        <div id="master-stat-count" class="status-indicator">Cargando snapshots maestros...</div>
      </div>

      <div id="masters-grid" class="grid"></div>

      <div class="pagination">
        <button id="btn-master-prev" class="btn">◀ Anterior</button>
        <span id="master-page-num" style="color: var(--text-muted); font-size: 0.9rem;">Página 1</span>
        <button id="btn-master-next" class="btn">Siguiente ▶</button>
      </div>
    </div>

    <!-- TAB 1: ATLAS 32 ZONAS & HUMAN-IN-THE-LOOP -->
    <div id="tab-atlas" class="tab-pane">
      <div class="filter-bar">
        <div class="form-group">
          <label for="model-select">Modelo Biométrico</label>
          <select id="model-select">
            <option value="all">Todos los Modelos</option>
          </select>
        </div>

        <div class="form-group">
          <label for="zone-select">Zona Anatómica (32 Zonas)</label>
          <select id="zone-select">
            <option value="all">Todas las 32 Zonas</option>
            <option value="cabeza">cabeza (craneal general)</option>
            <option value="cara">cara (rostro frontal)</option>
            <option value="ojo_izq">ojo_izq (espectro iris)</option>
            <option value="ojo_der">ojo_der (espectro iris)</option>
            <option value="nariz">nariz (puente nasal)</option>
            <option value="oreja_izq">oreja_izq (lóbulo)</option>
            <option value="oreja_der">oreja_der (lóbulo)</option>
            <option value="boca">boca y labios</option>
            <option value="cuello">cuello</option>
            <option value="pecho">pecho y torso superior</option>
            <option value="busto">busto (volumen mamario)</option>
            <option value="areola_izq">areola_izq (morfometría)</option>
            <option value="areola_der">areola_der (morfometría)</option>
            <option value="abdomen">abdomen</option>
            <option value="ombligo">ombligo</option>
            <option value="cintura">cintura</option>
            <option value="cadera">cadera y silueta</option>
            <option value="pelvis">pelvis</option>
            <option value="entrepierna">entrepierna / íntimo</option>
            <option value="pompi">pompi / glúteos</option>
            <option value="espalda">espalda superior</option>
            <option value="espalda_baja">espalda baja</option>
            <option value="brazo_izq">brazo_izq</option>
            <option value="brazo_der">brazo_der</option>
            <option value="mano_izq">mano_izq</option>
            <option value="mano_der">mano_der</option>
            <option value="muslo_izq">muslo_izq</option>
            <option value="muslo_der">muslo_der</option>
            <option value="rodilla_izq">rodilla_izq</option>
            <option value="rodilla_der">rodilla_der</option>
            <option value="pantorrilla_izq">pantorrilla_izq</option>
            <option value="pantorrilla_der">pantorrilla_der</option>
            <option value="pie_izq">pie_izq</option>
            <option value="pie_der">pie_der</option>
          </select>
        </div>

        <div class="form-group">
          <label for="status-select">Estado de Revisión</label>
          <select id="status-select">
            <option value="all">Todos los Estados</option>
            <option value="pending" selected>Pendientes de Revisión</option>
            <option value="verified">✅ Verificados</option>
            <option value="rejected">❌ Descartados</option>
          </select>
        </div>

        <div style="margin-left: auto; display: flex; align-items: flex-end;">
          <span id="stat-count" style="font-size: 0.85rem; color: var(--text-muted);">Cargando...</span>
        </div>
      </div>

      <div id="snapshots-grid" class="grid"></div>

      <div class="pagination">
        <button id="btn-prev-page" class="btn">◀ Anterior</button>
        <span id="page-indicator" style="font-size: 0.875rem; color: var(--text-muted);">Página 1</span>
        <button id="btn-next-page" class="btn">Siguiente ▶</button>
      </div>
    </div>

    <!-- TAB 2: VIDEOS & TIMELINE POR MINUTO -->
    <div id="tab-video" class="tab-pane">
      <div class="filter-bar">
        <div class="form-group" style="flex: 1; max-width: 450px;">
          <label for="video-select">Seleccionar Video Recolectado</label>
          <select id="video-select" style="width: 100%;">
            <option value="">-- Elige un video --</option>
          </select>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center; gap: 0.5rem;">
          <span id="video-stat" style="font-size: 0.85rem; color: var(--text-muted);">HTTP 206 Streaming Activo</span>
        </div>
      </div>

      <div class="video-view-container">
        <div class="player-box">
          <video id="video-player" controls preload="metadata">
            Tu navegador no soporta reproducción directa de video HTML5.
          </video>
          <div id="video-now-playing" style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted);"></div>
        </div>

        <div class="timeline-container">
          <div class="timeline-header">
            <div>
              <h3 style="font-size: 1.1rem; font-weight: 600;">⏱️ Línea de Tiempo Escénica (1 Snapshot por Minuto)</h3>
              <p style="font-size: 0.8rem; color: var(--text-muted);">Haz clic en cualquier minuto para saltar la reproducción a ese instante exacto.</p>
            </div>
            <span id="timeline-count" style="font-size: 0.85rem; color: #60a5fa;">0 escenas indexadas</span>
          </div>

          <div id="timeline-scroller" class="timeline-scroll">
            <!-- Timeline Cards inserted via JS -->
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: BUSCADOR CONTEXTUAL -->
    <div id="tab-search" class="tab-pane">
      <div class="filter-bar">
        <div class="form-group" style="flex: 2; min-width: 280px;">
          <label for="search-input">Búsqueda Contextual (FTS5 / Minuto / Descripción)</label>
          <input type="text" id="search-input" placeholder="ej: minuto 3, plano general, piel tan, busto, intermediate...">
        </div>
        <div class="form-group">
          <label for="search-model-select">Filtrar por Modelo</label>
          <select id="search-model-select">
            <option value="all">Todos los Modelos</option>
          </select>
        </div>
        <div style="display: flex; align-items: flex-end;">
          <button id="btn-search-exec" class="btn btn-primary">🔍 Buscar Escena</button>
        </div>
      </div>

      <div id="search-results-info" style="margin-bottom: 1rem; font-size: 0.9rem; color: var(--text-muted);">
        Escribe un término o minuto para buscar escenas en el catálogo forense.
      </div>

      <div id="search-results-grid" class="grid"></div>
    </div>

    <!-- TAB 4: MEMORIA COGNITIVA AGY -->
    <div id="tab-memory" class="tab-pane">
      <div class="filter-bar">
        <h3 style="font-size: 1rem; font-weight: 600; color: #a78bfa;">🧠 Memoria Cognitiva de Antigravity (Correcciones Humanas)</h3>
        <p style="font-size: 0.8rem; color: var(--text-muted);">
          Estas notas y categorizaciones de error son consultadas por AGY antes de cada cosecha para no repetir sesgos.
        </p>
      </div>
      <div id="agy-memory-list" style="display: flex; flex-direction: column; gap: 0.75rem;"></div>
    </div>

    <!-- TAB 5: COSECHA & ENTRENAR -->
    <div id="tab-harvest" class="tab-pane">
      <div class="filter-bar" style="flex-direction: column; align-items: stretch; gap: 1.5rem;">
        <div>
          <h3 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">🚀 Cosecha Omni-Source Bajo Demanda</h3>
          <p style="font-size: 0.85rem; color: var(--text-muted);">
            Ingesta desde Google Drive (multi-cuenta), Reels (Instagram, TikTok, X vía yt-dlp), Catálogos o Carpetas Locales.
          </p>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem;">
          <div class="form-group">
            <label>Modelo Destino</label>
            <input type="text" id="h-model" placeholder="ej: taanni_bc, uwxo, etc.">
          </div>
          <div class="form-group">
            <label>Origen de Medios</label>
            <select id="h-origin">
              <option value="auto">Auto-detección Inteligente</option>
              <option value="gdrive">Google Drive (rclone multi-remoto)</option>
              <option value="social">Redes Sociales / Reels (yt-dlp)</option>
              <option value="catalog">Catálogo Local / Datasets</option>
              <option value="local">Carpeta Local Específica</option>
            </select>
          </div>
          <div class="form-group">
            <label>URL o Ruta Específica (Opcional)</label>
            <input type="text" id="h-source-input" placeholder="ej: https://instagram.com/reel/... o /media/...">
          </div>
          <div class="form-group">
            <label>Situaciones x Video</label>
            <input type="number" id="h-situations" value="4" min="1" max="15">
          </div>
        </div>

        <div style="display: flex; gap: 1rem;">
          <button id="btn-trigger-harvest" class="btn btn-primary">⚡ Iniciar Cosecha</button>
          <button id="btn-trigger-training" class="btn btn-success">🎓 Re-entrenar Modelo Biométrico</button>
        </div>

        <div id="harvest-console" style="background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 1rem; font-family: monospace; font-size: 0.8rem; height: 200px; overflow-y: auto; color: #38bdf8;">
          Consola de ejecución en espera...
        </div>
      </div>
    </div>

  <!-- MODAL INSPECTOR: SNAPSHOT MAESTRO ➔ RECORTES ANATÓMICOS -->
  <div id="master-crop-modal" class="modal-overlay">
    <div class="modal-card" style="max-width: 1150px; width: 95%; height: 85vh; max-height: 850px; display: flex; flex-direction: column;">
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 0.8rem;">
        <div>
          <h3 id="modal-master-title" style="margin: 0; font-size: 1.15rem; color: #60a5fa;">📸 Inspección Jerárquica de Snapshot</h3>
          <span id="modal-master-meta" style="font-size: 0.8rem; color: var(--text-muted);">Foto original vs Recortes extraídos</span>
        </div>
        <button id="btn-close-master-modal" class="btn btn-secondary" style="padding: 0.3rem 0.8rem; font-size: 1rem;">✕ Cerrar</button>
      </div>

      <div style="display: flex; gap: 1.5rem; flex: 1; overflow: hidden; margin-top: 1rem;">
        <!-- Panel Izquierdo: Foto Original Completa -->
        <div style="flex: 1.1; display: flex; flex-direction: column; background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 0.8rem; overflow: hidden;">
          <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
            <span>🖼️ Fotograma Original Completo</span>
            <span id="modal-master-sharpness">Nitidez: --</span>
          </div>
          <div style="flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #020617; border-radius: 6px;">
            <img id="modal-master-img" src="" alt="Snapshot Original" style="max-width: 100%; max-height: 100%; object-fit: contain;">
          </div>
          <div id="modal-master-desc" style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem; max-height: 40px; overflow: hidden; text-overflow: ellipsis;"></div>
        </div>

        <!-- Panel Derecho: Recortes Anatómicos Extraídos -->
        <div style="flex: 1.4; display: flex; flex-direction: column; background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 0.8rem; overflow: hidden;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">✂️ Recortes Anatómicos Extraídos (<span id="modal-crops-count">0</span>)</span>
            <div style="display: flex; gap: 0.5rem;">
              <button id="btn-approve-all-crops" class="btn btn-primary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;">✔ Aprobar Todos</button>
            </div>
          </div>
          <div id="modal-crops-container" style="flex: 1; overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.8rem; padding-right: 0.3rem;">
            <!-- Se inyectan las tarjetas de cada recorte -->
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- REJECTION REASON MODAL -->
  <div id="rejection-modal" class="modal-overlay">
    <div class="modal-card">
      <h3 style="color: var(--danger); font-size: 1.1rem;">❌ Descartar Snapshot</h3>
      <p style="font-size: 0.85rem; color: var(--text-muted);">
        Indica el motivo para que el backend cognitivo de Antigravity (AGY) aprenda y refine futuras selecciones:
      </p>

      <div class="reason-chips">
        <span class="chip active" data-reason="Detección o corte anatómico erróneo">Detección errónea</span>
        <span class="chip" data-reason="Baja nitidez o artefacto de compresión">Baja nitidez</span>
        <span class="chip" data-reason="Mala iluminación o sobreexposición">Mala iluminación</span>
        <span class="chip" data-reason="Falso positivo de otra persona/modelo">Modelo ajena</span>
        <span class="chip" data-reason="Oclusión parcial / elemento que tapa el rasgo">Oclusión</span>
      </div>

      <div class="form-group">
        <label for="reject-note">Nota Opcional para AGY</label>
        <input type="text" id="reject-note" placeholder="ej: El recorte agarró la toalla en vez de la pierna">
      </div>

      <div style="display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 0.5rem;">
        <button id="modal-cancel-btn" class="btn">Cancelar</button>
        <button id="modal-confirm-reject-btn" class="btn btn-danger">Confirmar Descarte</button>
      </div>
    </div>
  </div>

  <script>
    // State
    let currentMasterModel = "all";
    let currentMasterPage = 1;
    let currentActiveMasterId = null;

    let currentModel = "all";
    let currentZone = "all";
    let currentStatus = "pending";
    let currentPage = 1;
    let pendingRejectId = null;
    let activeRejectReason = "Detección o corte anatómico erróneo";

    // Tab Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');

        if (tabId === 'tab-masters') {
          loadMasterSnapshots();
        } else if (tabId === 'tab-atlas') {
          loadSnapshots();
        } else if (tabId === 'tab-video') {
          loadVideosList();
        } else if (tabId === 'tab-memory') {
          loadAgyMemory();
        }
      });
    });

    // Load Models into Dropdowns
    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const models = data.models || [];
        const modelSelect = document.getElementById('model-select');
        const masterModelSelect = document.getElementById('master-model-select');
        const searchModelSelect = document.getElementById('search-model-select');

        modelSelect.innerHTML = '<option value="all">Todos los Modelos</option>';
        if (masterModelSelect) masterModelSelect.innerHTML = '<option value="all">Todos los Modelos</option>';
        searchModelSelect.innerHTML = '<option value="all">Todos los Modelos</option>';

        models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.model_name;
          const countNum = m.total !== undefined ? m.total : (m.count || 0);
          opt.textContent = `${m.model_name} (${countNum} recortes)`;
          modelSelect.appendChild(opt);

          if (masterModelSelect) {
            const optMaster = document.createElement('option');
            optMaster.value = m.model_name;
            optMaster.textContent = m.model_name;
            masterModelSelect.appendChild(optMaster);
          }

          const opt2 = document.createElement('option');
          opt2.value = m.model_name;
          opt2.textContent = m.model_name;
          searchModelSelect.appendChild(opt2);
        });
      } catch (err) {
        console.error("Error loading models:", err);
      }
    }

    // Load Master Snapshots
    async function loadMasterSnapshots() {
      const grid = document.getElementById('masters-grid');
      const statBox = document.getElementById('master-stat-count');
      grid.innerHTML = '<div style="color: var(--text-muted); padding: 2rem;">Cargando snapshots maestros...</div>';

      try {
        const res = await fetch(`/api/masters?model=${currentMasterModel}&page=${currentMasterPage}&limit=24`);
        const data = await res.json();

        statBox.textContent = `Total: ${data.total} fotos originales | Pág ${data.page}/${data.pages}`;
        document.getElementById('master-page-num').textContent = `Página ${data.page} de ${data.pages}`;

        grid.innerHTML = '';
        if (!data.masters || data.masters.length === 0) {
          grid.innerHTML = '<div style="color: var(--text-muted); grid-column: 1/-1; padding: 3rem; text-align: center;">No hay snapshots maestros disponibles.</div>';
          return;
        }

        data.masters.forEach(item => {
          const card = document.createElement('div');
          card.className = `card ${item.status || 'pending'}`;
          card.style.cursor = 'pointer';
          card.innerHTML = `
            <div class="card-img-wrapper">
              <img class="card-img" src="/api/image/${item.id}" alt="${item.image_filename}">
              <span class="zone-badge" style="background: rgba(14, 165, 233, 0.85);">📸 ${item.timestamp_str || 'Escena'}</span>
              <span class="score-badge">★ ${(item.studio_score || 0).toFixed(1)}</span>
            </div>
            <div class="card-body">
              <div class="card-title">
                <span>${item.model_name}</span>
                <span class="card-meta">Laplacian: ${(item.sharpness || 0).toFixed(1)}</span>
              </div>
              <p class="card-desc">${item.source_video || 'Video'}</p>
              <div style="margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 600;">✂️ ${item.crops_count || 0} recortes</span>
                <button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;">🔍 Inspeccionar</button>
              </div>
            </div>
          `;
          card.addEventListener('click', () => inspectMasterSnapshot(item.id));
          grid.appendChild(card);
        });
      } catch (err) {
        console.error("Error loading master snapshots:", err);
        grid.innerHTML = `<div style="color: var(--danger); padding: 2rem;">Error cargando snapshots maestros: ${err.message}</div>`;
      }
    }

    // Inspect Master Snapshot & Its Derived Crops
    async function inspectMasterSnapshot(masterId) {
      currentActiveMasterId = masterId;
      const modal = document.getElementById('master-crop-modal');
      const container = document.getElementById('modal-crops-container');
      container.innerHTML = '<div style="color: var(--text-muted); padding: 1rem;">Cargando recortes anatómicos derivados...</div>';
      modal.classList.add('active');

      try {
        const res = await fetch(`/api/master/${masterId}/crops`);
        const data = await res.json();
        if (!data.success) {
          alert("Error: " + data.error);
          return;
        }

        const m = data.master;
        document.getElementById('modal-master-title').textContent = `📸 Inspección: ${m.model_name} (${m.timestamp_str || ''})`;
        document.getElementById('modal-master-meta').textContent = `${m.source_video || ''} | Foto ID #${m.id}`;
        document.getElementById('modal-master-sharpness').textContent = `Nitidez: ${(m.sharpness || 0).toFixed(1)} | Calidad: ★ ${(m.studio_score || 0).toFixed(1)}`;
        document.getElementById('modal-master-img').src = `/api/image/${m.id}`;
        document.getElementById('modal-master-desc').textContent = m.description || '';
        document.getElementById('modal-crops-count').textContent = data.crops_count;

        container.innerHTML = '';
        if (!data.crops || data.crops.length === 0) {
          container.innerHTML = '<div style="color: var(--text-muted); padding: 2rem; grid-column: 1/-1; text-align: center;">No hay recortes anatómicos derivados para este snapshot.</div>';
          return;
        }

        data.crops.forEach(c => {
          const card = document.createElement('div');
          card.className = `card ${c.status || 'pending'}`;
          card.id = `modal-crop-${c.id}`;
          card.style.margin = '0';
          card.innerHTML = `
            <div class="card-img-wrapper" style="height: 120px;">
              <img class="card-img" src="/api/image/${c.id}" alt="${c.image_filename}">
              <span class="zone-badge">${c.zone_type}</span>
              <span class="score-badge" style="font-size: 0.65rem;">★ ${(c.studio_score || 0).toFixed(1)}</span>
            </div>
            <div class="card-body" style="padding: 0.5rem;">
              <div style="font-size: 0.75rem; color: #94a3b8; display: flex; justify-content: space-between;">
                <span>${c.zone_type}</span>
                <span>Lap: ${(c.sharpness || 0).toFixed(1)}</span>
              </div>
              <div class="card-actions" style="margin-top: 0.4rem;">
                <button class="btn btn-sm btn-verify" style="font-size: 0.7rem; padding: 0.2rem 0.4rem;" onclick="verifyCropInModal(${c.id}, event)">✔</button>
                <button class="btn btn-sm btn-reject" style="font-size: 0.7rem; padding: 0.2rem 0.4rem;" onclick="rejectCropInModal(${c.id}, event)">✕</button>
              </div>
            </div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        console.error("Error inspecting master:", err);
      }
    }

    // Modal Crop Actions
    async function verifyCropInModal(cropId, e) {
      if (e) e.stopPropagation();
      try {
        const res = await fetch(`/api/verify/${cropId}`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          const card = document.getElementById(`modal-crop-${cropId}`);
          if (card) {
            card.className = 'card verified';
          }
        }
      } catch (err) {
        console.error(err);
      }
    }

    async function rejectCropInModal(cropId, e) {
      if (e) e.stopPropagation();
      try {
        const res = await fetch(`/api/reject/${cropId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note: 'Descarte visual desde inspector jerárquico' })
        });
        const data = await res.json();
        if (data.success) {
          const card = document.getElementById(`modal-crop-${cropId}`);
          if (card) {
            card.className = 'card rejected';
          }
        }
      } catch (err) {
        console.error(err);
      }
    }

    document.getElementById('btn-close-master-modal').addEventListener('click', () => {
      document.getElementById('master-crop-modal').classList.remove('active');
    });

    document.getElementById('btn-approve-all-crops').addEventListener('click', async () => {
      const cards = document.querySelectorAll('#modal-crops-container .card');
      for (const card of cards) {
        const cid = card.id.replace('modal-crop-', '');
        await fetch(`/api/verify/${cid}`, { method: 'POST' });
        card.className = 'card verified';
      }
    });

    // Load Snapshots for Atlas Grid
    async function loadSnapshots() {
      const grid = document.getElementById('snapshots-grid');
      grid.innerHTML = '<div style="color: var(--text-muted); padding: 2rem;">Cargando snapshots...</div>';

      try {
        const url = `/api/snapshots?model=${currentModel}&zone=${currentZone}&status=${currentStatus}&page=${currentPage}&page_size=24`;
        const res = await fetch(url);
        const data = await res.json();

        document.getElementById('stat-count').textContent = `Total: ${data.total} elementos | Pág ${data.page}/${data.total_pages}`;
        document.getElementById('page-indicator').textContent = `Página ${data.page} de ${data.total_pages}`;

        grid.innerHTML = '';
        if (!data.items || data.items.length === 0) {
          grid.innerHTML = '<div style="color: var(--text-muted); grid-column: 1/-1; padding: 3rem; text-align: center;">No hay snapshots para este filtro.</div>';
          return;
        }

        data.items.forEach(item => {
          const card = document.createElement('div');
          card.className = `card ${item.status || 'pending'}`;
          card.id = `snap-card-${item.id}`;

          card.innerHTML = `
            <div class="card-img-wrapper">
              <img class="card-img" src="/api/image/${item.id}" alt="${item.image_filename}">
              <span class="zone-badge">${item.zone_type}</span>
              <span class="score-badge">★ ${(item.studio_score || 0).toFixed(1)}</span>
            </div>
            <div class="card-body">
              <div class="card-title">
                <span>${item.model_name}</span>
                <span style="font-size: 0.75rem; color: #60a5fa;">${item.timestamp_str || ''}</span>
              </div>
              <div class="card-desc" title="${item.description || ''}">${item.description || 'Sin descripción'}</div>
              <div class="card-actions">
                <button class="btn btn-success" onclick="verifySnapshot(${item.id})">✅ Verificado</button>
                <button class="btn btn-danger" onclick="openRejectModal(${item.id})">❌ Descartar</button>
              </div>
            </div>
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        grid.innerHTML = `<div style="color: var(--danger); padding: 2rem;">Error cargando snapshots: ${err.message}</div>`;
      }
    }

    // Verify action
    async function verifySnapshot(id) {
      try {
        const res = await fetch(`/api/verify/${id}`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          const card = document.getElementById(`snap-card-${id}`);
          if (card) {
            card.className = 'card verified';
            setTimeout(() => { if (currentStatus === 'pending') card.remove(); }, 500);
          }
        }
      } catch (err) {
        alert("Error verificando: " + err.message);
      }
    }

    // Modal Rejection Handling
    function openRejectModal(id) {
      pendingRejectId = id;
      document.getElementById('reject-note').value = '';
      document.getElementById('rejection-modal').classList.add('active');
    }

    document.getElementById('modal-cancel-btn').addEventListener('click', () => {
      document.getElementById('rejection-modal').classList.remove('active');
      pendingRejectId = null;
    });

    document.querySelectorAll('.reason-chips .chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.reason-chips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        activeRejectReason = chip.getAttribute('data-reason');
      });
    });

    document.getElementById('modal-confirm-reject-btn').addEventListener('click', async () => {
      if (!pendingRejectId) return;
      const note = document.getElementById('reject-note').value.trim();

      try {
        const res = await fetch(`/api/reject/${pendingRejectId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            error_category: activeRejectReason,
            note: note
          })
        });
        const data = await res.json();
        if (data.success) {
          const card = document.getElementById(`snap-card-${pendingRejectId}`);
          if (card) {
            card.className = 'card rejected';
            setTimeout(() => { if (currentStatus === 'pending') card.remove(); }, 500);
          }
        }
      } catch (err) {
        alert("Error al descartar: " + err.message);
      } finally {
        document.getElementById('rejection-modal').classList.remove('active');
        pendingRejectId = null;
      }
    });

    // ==========================================
    // Video Player & Minute Timeline Logic
    // ==========================================
    async function loadVideosList() {
      const select = document.getElementById('video-select');
      try {
        const res = await fetch(`/api/videos?model=${currentModel}`);
        const data = await res.json();
        select.innerHTML = '<option value="">-- Elige un video --</option>';

        (data.videos || []).forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.filename;
          opt.textContent = `${v.filename} (${v.scenes_count} min indexados)`;
          opt.setAttribute('data-path', v.path || '');
          select.appendChild(opt);
        });

        if (data.videos && data.videos.length > 0) {
          select.selectedIndex = 1;
          onVideoSelected();
        }
      } catch (err) {
        console.error("Error loading videos:", err);
      }
    }

    document.getElementById('video-select').addEventListener('change', onVideoSelected);

    function onVideoSelected() {
      const select = document.getElementById('video-select');
      const filename = select.value;
      if (!filename) return;

      const player = document.getElementById('video-player');
      const streamUrl = `/api/video/stream?filename=${encodeURIComponent(filename)}`;
      player.src = streamUrl;
      document.getElementById('video-now-playing').textContent = `Reproduciendo: ${filename}`;

      loadTimelineForVideo(filename);
    }

    async function loadTimelineForVideo(filename) {
      const scroller = document.getElementById('timeline-scroller');
      scroller.innerHTML = '<div style="color: var(--text-muted); padding: 1rem;">Cargando escenas...</div>';

      try {
        const res = await fetch(`/api/video/scenes?video=${encodeURIComponent(filename)}`);
        const data = await res.json();
        const scenes = data.scenes || [];

        document.getElementById('timeline-count').textContent = `${scenes.length} escenas por minuto`;
        scroller.innerHTML = '';

        if (scenes.length === 0) {
          scroller.innerHTML = '<div style="color: var(--text-muted); padding: 1rem;">No hay escenas por minuto extraídas aún para este video.</div>';
          return;
        }

        scenes.forEach(s => {
          const card = document.createElement('div');
          card.className = 'timeline-card';
          card.id = `timeline-card-${s.id}`;
          card.innerHTML = `
            <img class="timeline-thumb" src="/api/video/scene_image/${s.id}" alt="Minuto ${s.minute_index}">
            <div class="timeline-body">
              <div class="timeline-time">
                <span>Min ${s.minute_index}</span>
                <span>${s.timestamp_str}</span>
              </div>
              <div class="timeline-desc" title="${s.scene_description}">${s.scene_description}</div>
            </div>
          `;

          card.addEventListener('click', () => {
            document.querySelectorAll('.timeline-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            const player = document.getElementById('video-player');
            player.currentTime = s.timestamp_sec;
            player.play();
          });

          scroller.appendChild(card);
        });
      } catch (err) {
        scroller.innerHTML = `<div style="color: var(--danger); padding: 1rem;">Error cargando timeline: ${err.message}</div>`;
      }
    }

    // ==========================================
    // Contextual Search Logic
    // ==========================================
    async function execSearch() {
      const q = document.getElementById('search-input').value.trim();
      const model = document.getElementById('search-model-select').value;
      const grid = document.getElementById('search-results-grid');
      const info = document.getElementById('search-results-info');

      if (!q) {
        info.textContent = "Por favor ingresa un término de búsqueda (ej: 'minuto 3', 'Brown', 'Plano Medio').";
        return;
      }

      info.textContent = `Buscando "${q}"...`;
      grid.innerHTML = '';

      try {
        const url = `/api/search?q=${encodeURIComponent(q)}&model=${encodeURIComponent(model)}`;
        const res = await fetch(url);
        const data = await res.json();
        const results = data.results || [];

        info.textContent = `Se encontraron ${results.length} coincidencias para "${q}".`;
        if (results.length === 0) {
          grid.innerHTML = '<div style="color: var(--text-muted); grid-column: 1/-1; padding: 2rem;">No se encontraron escenas que coincidan con la búsqueda.</div>';
          return;
        }

        results.forEach(r => {
          const card = document.createElement('div');
          card.className = 'card';
          card.innerHTML = `
            <div class="card-img-wrapper" style="height: 180px;">
              <img class="card-img" src="/api/video/scene_image/${r.id}" alt="Escena ${r.timestamp_str}">
              <span class="zone-badge">Minuto ${r.minute_index}</span>
              <span class="score-badge">${r.timestamp_str}</span>
            </div>
            <div class="card-body">
              <div class="card-title">
                <span>${r.model_name}</span>
                <span style="font-size: 0.75rem; color: #38bdf8;">${r.video_filename}</span>
              </div>
              <div class="card-desc" style="max-height: 4rem;">${r.scene_description}</div>
              <div class="card-actions">
                <button class="btn btn-primary" onclick="jumpToSceneInVideo('${r.video_filename}', ${r.timestamp_sec})">▶ Reproducir Escena</button>
              </div>
            </div>
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        info.textContent = `Error en búsqueda: ${err.message}`;
      }
    }

    document.getElementById('btn-search-exec').addEventListener('click', execSearch);
    document.getElementById('search-input').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') execSearch();
    });

    window.jumpToSceneInVideo = function(videoFilename, timestampSec) {
      // Switch to video tab
      document.querySelector('[data-tab="tab-video"]').click();
      const select = document.getElementById('video-select');
      select.value = videoFilename;
      onVideoSelected();

      setTimeout(() => {
        const player = document.getElementById('video-player');
        player.currentTime = timestampSec;
        player.play();
      }, 300);
    };

    // ==========================================
    // AGY Memory View
    // ==========================================
    async function loadAgyMemory() {
      const container = document.getElementById('agy-memory-list');
      container.innerHTML = '<div style="color: var(--text-muted); padding: 1rem;">Cargando memorias de aprendizaje cognitivo...</div>';

      try {
        const res = await fetch(`/api/agy/memory?model=${currentModel}`);
        const data = await res.json();
        const memories = data.memories || [];
        container.innerHTML = '';

        if (memories.length === 0) {
          container.innerHTML = '<div style="color: var(--text-muted); padding: 1.5rem;">Aún no hay correcciones registradas. Al descartar recortes y añadir notas, aparecerán aquí para guiar a AGY.</div>';
          return;
        }

        memories.forEach(m => {
          const item = document.createElement('div');
          item.style = "background: #111827; border: 1px solid #1f2937; border-left: 4px solid #8b5cf6; padding: 1rem; border-radius: 6px; display: flex; flex-direction: column; gap: 0.35rem;";
          item.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
              <span style="font-weight: 700; color: #c4b5fd;">${m.model_name} • Zona: ${m.zone_type}</span>
              <span style="color: var(--text-muted); font-size: 0.75rem;">${m.created_at || ''}</span>
            </div>
            <div style="font-size: 0.85rem; color: #f87171;"><strong>Motivo:</strong> ${m.error_category}</div>
            <div style="font-size: 0.85rem; color: var(--text);"><strong>Nota de corrección:</strong> ${m.user_note || '(Sin nota adicional)'}</div>
          `;
          container.appendChild(item);
        });
      } catch (err) {
        container.innerHTML = `<div style="color: var(--danger); padding: 1rem;">Error cargando memoria: ${err.message}</div>`;
      }
    }

    // ==========================================
    // Harvest & Training Triggers
    // ==========================================
    document.getElementById('btn-trigger-harvest').addEventListener('click', async () => {
      const model = document.getElementById('h-model').value.trim();
      if (!model) { alert("Ingresa el nombre del modelo."); return; }

      const origin = document.getElementById('h-origin').value;
      const sourceInput = document.getElementById('h-source-input').value.trim();
      const situations = parseInt(document.getElementById('h-situations').value) || 4;

      const consoleBox = document.getElementById('harvest-console');
      consoleBox.textContent = `Iniciando cosecha para ${model} desde ${origin}...\n`;

      try {
        const res = await fetch('/api/harvest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: model,
            origin: origin,
            source_input: sourceInput,
            situations: situations
          })
        });
        const data = await res.json();
        if (data.success) {
          consoleBox.textContent += `Tarea enviada con éxito (Job ID: ${data.job_id}). Monitoreando en segundo plano...\n`;
          pollHarvestJob(data.job_id);
        }
      } catch (err) {
        consoleBox.textContent += `Error iniciando cosecha: ${err.message}\n`;
      }
    });

    async function pollHarvestJob(jobId) {
      const consoleBox = document.getElementById('harvest-console');
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`/api/harvest/status/${jobId}`);
          const job = await res.json();
          if (job.log) {
            consoleBox.textContent = job.log.join(String.fromCharCode(10));
            consoleBox.scrollTop = consoleBox.scrollHeight;
          }
          if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(interval);
            consoleBox.textContent += `\n=== Cosecha finalizada con estado: ${job.status} ===\n`;
            loadModels();
            loadSnapshots();
          }
        } catch (e) {
          clearInterval(interval);
        }
      }, 3000);
    }

    document.getElementById('btn-trigger-training').addEventListener('click', async () => {
      const model = document.getElementById('h-model').value.trim() || currentModel;
      if (!model || model === 'all') {
        alert("Selecciona o ingresa un modelo específico para re-entrenar.");
        return;
      }
      const consoleBox = document.getElementById('harvest-console');
      consoleBox.textContent = `Disparando re-entrenamiento biométrico para ${model}...\n`;

      try {
        const res = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: model })
        });
        const data = await res.json();
        consoleBox.textContent += JSON.stringify(data, null, 2);
      } catch (err) {
        consoleBox.textContent += `Error en entrenamiento: ${err.message}\n`;
      }
    });

    // Event Listeners for Filters
    document.getElementById('model-select').addEventListener('change', (e) => {
      currentModel = e.target.value;
      currentPage = 1;
      loadSnapshots();
    });

    document.getElementById('zone-select').addEventListener('change', (e) => {
      currentZone = e.target.value;
      currentPage = 1;
      loadSnapshots();
    });

    document.getElementById('status-select').addEventListener('change', (e) => {
      currentStatus = e.target.value;
      currentPage = 1;
      loadSnapshots();
    });

    document.getElementById('btn-prev-page').addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        loadSnapshots();
      }
    });

    document.getElementById('btn-next-page').addEventListener('click', () => {
      currentPage++;
      loadSnapshots();
    });

    const mModelSelect = document.getElementById('master-model-select');
    if (mModelSelect) {
      mModelSelect.addEventListener('change', (e) => {
        currentMasterModel = e.target.value;
        currentMasterPage = 1;
        loadMasterSnapshots();
      });
    }

    const btnMPrev = document.getElementById('btn-master-prev');
    if (btnMPrev) {
      btnMPrev.addEventListener('click', () => {
        if (currentMasterPage > 1) {
          currentMasterPage--;
          loadMasterSnapshots();
        }
      });
    }

    const btnMNext = document.getElementById('btn-master-next');
    if (btnMNext) {
      btnMNext.addEventListener('click', () => {
        currentMasterPage++;
        loadMasterSnapshots();
      });
    }

    document.getElementById('btn-refresh').addEventListener('click', () => {
      loadModels();
      loadMasterSnapshots();
      loadSnapshots();
    });

    // Init
    loadModels().then(() => {
      loadMasterSnapshots();
      loadSnapshots();
    });
  </script>
</body>
</html>
"""


def create_app(db_path: Path) -> web.Application:
    studio = WebStudioApp(db_path)
    app = web.Application()

    # Dashboard & Assets
    app.router.add_get("/", studio.render_dashboard)

    # REST APIs: Models & Snapshots Triage
    app.router.add_get("/api/models", studio.get_models)
    app.router.add_get("/api/masters", studio.get_master_snapshots)
    app.router.add_get("/api/master/{id}/crops", studio.get_master_crops)
    app.router.add_get("/api/snapshots", studio.get_snapshots)
    app.router.add_get("/api/image/{id}", studio.get_image)
    app.router.add_post("/api/verify/{id}", studio.verify_snapshot)
    app.router.add_post("/api/reject/{id}", studio.reject_snapshot)
    app.router.add_post("/api/feedback/{id}", studio.record_feedback)
    app.router.add_get("/api/agy/memory", studio.get_agy_memory)

    # REST APIs: Video Streaming & Timeline Scenes
    app.router.add_get("/api/videos", studio.get_videos)
    app.router.add_get("/api/video/stream", studio.stream_video)
    app.router.add_get("/api/video/scenes", studio.get_scenes)
    app.router.add_get("/api/video/scene_image/{id}", studio.get_scene_image)
    app.router.add_get("/api/search", studio.search_scenes)

    # REST APIs: Omni-Source Harvest & Training
    app.router.add_post("/api/harvest", studio.trigger_harvest)
    app.router.add_get("/api/harvest/status/{job_id}", studio.get_harvest_status)
    app.router.add_post("/api/train", studio.trigger_training)

    return app


def main():
    parser = argparse.ArgumentParser(description="Antigravity Active Learning Studio Web UI")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite catalog database")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host bind address (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=8585, help="HTTP port (default: 8585)")
    args = parser.parse_args()

    db_path = Path(os.path.expanduser(args.db)).resolve()
    print(f"🚀 Iniciando Antigravity Active Studio Web Server:")
    print(f"   🗄️ Base de datos SQLite: {db_path}")
    print(f"   🌐 URL local: http://localhost:{args.port}")

    app = create_app(db_path)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
