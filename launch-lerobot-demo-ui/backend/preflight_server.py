"""
Preflight Camera Calibration Server

A lightweight FastAPI server that helps users:
1. Discover available video devices
2. Preview live snapshots from each camera
3. Assign camera roles (front / wrist)
4. Detect available robot serial ports
5. Save the mapping to config.py

IMPORTANT: This server does NOT import OpenCV or any GPU libraries.
All camera operations are delegated to _camera_worker.py which runs
in a separate subprocess. This prevents CUDA driver corruption that
would break the main inference process (eval_act_safe.py).

Snapshot delivery uses a PERSISTENT worker process ("serve" mode) that
keeps cameras open and writes JPEG frames to a shared directory.  This
avoids the open/close churn that causes green frames on YUYV cameras
(e.g. RealSense RGB on /dev/video5).

Usage:
    uvicorn preflight_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.py"
WORKER_SCRIPT = Path(__file__).parent / "_camera_worker.py"
DETECT_TIMEOUT = 60      # seconds to wait for camera detection
SNAPSHOT_TIMEOUT = 30    # seconds to wait for camera snapshot (fallback one-shot)

# Directory where the persistent worker writes JPEG frames
SERVE_FRAME_DIR = "/tmp/lerobot_preflight_frames"

# ── Persistent worker state ───────────────────────────────────────────────
_serve_process: subprocess.Popen | None = None
_serve_devices: list[str] = []   # devices currently being served


def _start_serve_worker(devices: list[str]) -> None:
    """Start (or restart) the persistent camera worker for the given devices."""
    global _serve_process, _serve_devices

    # Kill existing worker if running
    _stop_serve_worker()

    if not devices:
        return

    os.makedirs(SERVE_FRAME_DIR, exist_ok=True)

    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "serve", SERVE_FRAME_DIR,
    ] + devices

    _serve_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    _serve_devices = list(devices)

    # Wait until the worker has written at least one good frame for each device,
    # or until a generous timeout expires.  This prevents the frontend from
    # seeing green frames during the YUYV warmup period.
    deadline = time.monotonic() + 12.0  # up to 12s for slow YUYV cameras
    while time.monotonic() < deadline:
        time.sleep(0.5)
        # Check if all devices have a fresh .jpg file
        all_ready = True
        for dev in devices:
            fname = dev.replace("/dev/", "").replace("/", "_")
            jpg_path = os.path.join(SERVE_FRAME_DIR, f"{fname}.jpg")
            if not os.path.exists(jpg_path):
                all_ready = False
                break
        if all_ready:
            break
    # Even if not all cameras are ready, we continue — the serve worker
    # will keep trying and the fallback one-shot path is still available.


def _stop_serve_worker() -> None:
    """Stop the persistent camera worker if running."""
    global _serve_process, _serve_devices
    if _serve_process is not None:
        try:
            _serve_process.terminate()
            _serve_process.wait(timeout=5)
        except Exception:
            try:
                _serve_process.kill()
            except Exception:
                pass
        _serve_process = None
        _serve_devices = []


def _read_served_snapshot(device: str) -> bytes | None:
    """Read the latest JPEG frame written by the serve worker.
    Returns None if no frame is available yet.
    """
    fname = device.replace("/dev/", "").replace("/", "_")
    jpg_path = os.path.join(SERVE_FRAME_DIR, f"{fname}.jpg")

    if not os.path.exists(jpg_path):
        return None

    # Check freshness — if the file is older than 10s, the worker may be stuck
    try:
        age = time.time() - os.path.getmtime(jpg_path)
        if age > 10.0:
            return None
    except OSError:
        return None

    try:
        with open(jpg_path, "rb") as f:
            data = f.read()
        return data if data else None
    except OSError:
        return None


def _read_served_meta(device: str) -> dict | None:
    """Read the metadata file for a device from the serve worker."""
    fname = device.replace("/dev/", "").replace("/", "_")
    meta_path = os.path.join(SERVE_FRAME_DIR, f"{fname}.meta")
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _stop_serve_worker()


app = FastAPI(title="LeRobot Preflight Check", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Subprocess helpers (one-shot, for detect) ─────────────────────────────

def _worker_detect_cameras() -> list[dict[str, Any]]:
    """Run _camera_worker.py detect in a subprocess, return camera list."""
    result = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT), "detect"],
        capture_output=True,
        timeout=DETECT_TIMEOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Camera detection failed: {stderr}")

    stdout = result.stdout.decode(errors="replace").strip()
    if not stdout:
        return []
    return json.loads(stdout)


def _worker_capture_snapshot_oneshot(device: str) -> bytes:
    """Fallback: one-shot snapshot via subprocess (used if serve worker not running)."""
    try:
        result = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), "snapshot", device],
            capture_output=True,
            timeout=SNAPSHOT_TIMEOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Snapshot timeout for {device} after {SNAPSHOT_TIMEOUT}s.")
    
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Snapshot failed for {device}: {stderr}")

    jpeg_bytes = result.stdout
    if not jpeg_bytes:
        raise RuntimeError(f"Empty snapshot from {device}")

    return jpeg_bytes


def _get_snapshot(device: str) -> bytes:
    """Get a snapshot — prefer the persistent serve worker, fall back to one-shot."""
    # Try the serve worker first
    if _serve_process is not None and _serve_process.poll() is None:
        # If the worker is running, wait a bit for the frame to appear
        # rather than immediately falling back to one-shot (which would
        # also produce a green frame on YUYV cameras).
        for _ in range(10):  # up to 5 seconds
            data = _read_served_snapshot(device)
            if data:
                return data
            time.sleep(0.5)

    # Fall back to one-shot subprocess (serve worker not running or device not served)
    return _worker_capture_snapshot_oneshot(device)


# ── Non-camera helpers ─────────────────────────────────────────────────────

def _detect_serial_ports() -> list[dict[str, str]]:
    """Detect available serial ports (robot motor bus connections)."""
    ports: list[dict[str, str]] = []
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            ports.append({
                "device": p.device,
                "description": p.description or "",
                "manufacturer": p.manufacturer or "",
                "hwid": p.hwid or "",
            })
    except ImportError:
        for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
            for dev in sorted(glob.glob(pattern)):
                ports.append({"device": dev, "description": "", "manufacturer": "", "hwid": ""})
    return ports


def _read_current_config() -> dict[str, Any]:
    """Parse the current config.py and return camera/port values.

    Supports both formats:
      - Top-level variables:  CAMERAS = "front:/dev/video4,wrist:/dev/video6"
      - Dict values:          "cameras": "front:/dev/video4,wrist:/dev/video6"
    """
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text()

    # Try top-level CAMERAS = "..." first, then fall back to dict style
    m = re.search(r'^CAMERAS\s*=\s*"([^"]*)"', content, re.MULTILINE)
    if not m:
    m = re.search(r'"cameras"\s*:\s*"([^"]*)"', content)
    cameras_str = m.group(1) if m else ""

    m2 = re.search(r'^ROBOT_PORT\s*=\s*"([^"]*)"', content, re.MULTILINE)
    if not m2:
    m2 = re.search(r'"robot_port"\s*:\s*"([^"]*)"', content)
    robot_port = m2.group(1) if m2 else ""

    camera_map = {}
    if cameras_str:
        for pair in cameras_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                role, dev = pair.split(":", 1)
                camera_map[role.strip()] = dev.strip()
    return {
        "cameras": camera_map,
        "cameras_raw": cameras_str,
        "robot_port": robot_port,
    }


def _save_config(cameras_str: str, robot_port: str | None = None) -> None:
    """Update config.py with new camera mapping and optionally robot port.

    Supports both formats:
      - Top-level variables:  CAMERAS = "..."  /  ROBOT_PORT = "..."
      - Dict values:          "cameras": "..."  /  "robot_port": "..."
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    content = CONFIG_PATH.read_text()

    # Update top-level CAMERAS = "..." if present, else fall back to dict style
    if re.search(r'^CAMERAS\s*=\s*"', content, re.MULTILINE):
        content = re.sub(
            r'^(CAMERAS\s*=\s*)"([^"]*)"',
            f'\\1"{cameras_str}"',
            content,
            flags=re.MULTILINE,
        )
    else:
    content = re.sub(
        r'("cameras"\s*:\s*)"([^"]*)"',
        f'\\1"{cameras_str}"',
        content,
    )

    if robot_port:
        if re.search(r'^ROBOT_PORT\s*=\s*"', content, re.MULTILINE):
            content = re.sub(
                r'^(ROBOT_PORT\s*=\s*)"([^"]*)"',
                f'\\1"{robot_port}"',
                content,
                flags=re.MULTILINE,
            )
        else:
        content = re.sub(
            r'("robot_port"\s*:\s*)"([^"]*)"',
            f'\\1"{robot_port}"',
            content,
        )

    CONFIG_PATH.write_text(content)


# ── API Models ─────────────────────────────────────────────────────────────

class CameraAssignment(BaseModel):
    role: str    # "front" or "wrist"
    device: str  # e.g. "/dev/video4"


class SaveConfigRequest(BaseModel):
    cameras: list[CameraAssignment]
    robot_port: str | None = None


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/preflight/detect-cameras")
async def detect_cameras():
    """
    Detect all usable video capture devices.
    Camera work runs in a separate subprocess (_camera_worker.py).
    After detection, starts a persistent worker to keep cameras open.
    """
    loop = asyncio.get_event_loop()
    try:
        cameras = await loop.run_in_executor(None, _worker_detect_cameras)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Start persistent serve worker for all detected devices
    if cameras:
        detected_devices = [c["device"] for c in cameras]
        await loop.run_in_executor(None, _start_serve_worker, detected_devices)

    return {"cameras": cameras, "count": len(cameras)}


@app.get("/api/preflight/snapshot/{device_path:path}")
async def get_snapshot(device_path: str):
    """
    Return the latest snapshot from a specific video device.
    Uses the persistent serve worker (fast, no green frames).
    Falls back to one-shot subprocess if serve worker is not running.
    """
    device = f"/{device_path}"
    if not os.path.exists(device):
        raise HTTPException(status_code=404, detail=f"Device {device} not found")

    loop = asyncio.get_event_loop()
    try:
        jpeg_bytes = await loop.run_in_executor(None, _get_snapshot, device)
    except RuntimeError as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            raise HTTPException(status_code=408, detail=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/preflight/detect-ports")
async def detect_ports():
    """Detect available serial ports for robot connection."""
    loop = asyncio.get_event_loop()
    ports = await loop.run_in_executor(None, _detect_serial_ports)
    return {"ports": ports}


@app.get("/api/preflight/current-config")
async def get_current_config():
    """Return the current camera and port configuration from config.py."""
    config = _read_current_config()
    return {"config": config}


@app.post("/api/preflight/save-config")
async def save_config(req: SaveConfigRequest):
    """Save camera assignments and robot port to config.py."""
    parts = [f"{c.role}:{c.device}" for c in req.cameras]
    cameras_str = ",".join(parts)

    try:
        _save_config(cameras_str, req.robot_port)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    return {
        "status": "ok",
        "message": "Configuration saved successfully",
        "cameras": cameras_str,
        "robot_port": req.robot_port,
    }


@app.post("/api/preflight/launch-control")
async def launch_control():
    """
    Switch from preflight mode to main robot control mode.
    1. Stops the persistent camera worker
    2. Spawns main_robot:app uvicorn process in background (detached)
    3. Schedules self-termination of this preflight server after 1.5s
    """
    # Stop the serve worker so cameras are released for main_robot
    _stop_serve_worker()

    backend_dir = Path(__file__).parent
    conda_sh = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")

    # Explicitly unset CUDA_VISIBLE_DEVICES so main_robot sees the GPU
    cmd = (
        f"unset CUDA_VISIBLE_DEVICES && "
        f"source {conda_sh} && conda activate lerobot && "
        f"cd {backend_dir} && "
        f"exec uvicorn main_robot:app --host 0.0.0.0 --port 8000"
    )

    # Clean environment — remove CUDA_VISIBLE_DEVICES if set
    clean_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}

    subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=clean_env,
        start_new_session=True,
    )

    async def self_terminate():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(self_terminate())

    return {"status": "ok", "message": "Switching to control mode — please wait ~30s for warmup"}


@app.get("/api/preflight/health")
async def health():
    return {"status": "ok", "service": "preflight"}


@app.get("/api/health")
async def health_generic():
    """Generic health check — used by frontend to detect backend type."""
    return {"status": "ok", "service": "preflight"}
