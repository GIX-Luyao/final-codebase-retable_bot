"""
Preflight Camera Calibration Server

A lightweight FastAPI server that helps users:
1. Discover available video devices
2. Preview live snapshots from each camera
3. Assign camera roles (front / wrist)
4. Detect available robot serial ports
5. Save the mapping to config.py

Usage:
    uvicorn preflight_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import glob
import os
import platform
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.py"
SNAPSHOT_TIMEOUT = 3.0  # seconds to wait for a camera frame
CACHE_TTL = 0.4  # seconds to cache a snapshot per device


# ── Snapshot cache (avoids re-opening cameras on rapid polls) ──────────────
_snapshot_cache: dict[str, tuple[float, bytes]] = {}


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    yield
    _snapshot_cache.clear()


app = FastAPI(title="LeRobot Preflight Check", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _detect_video_devices() -> list[dict[str, Any]]:
    """Scan /dev/video* and return metadata for each working camera."""
    cameras: list[dict[str, Any]] = []

    if platform.system() == "Linux":
        paths = sorted(glob.glob("/dev/video*"))
    else:
        paths = [str(i) for i in range(20)]

    for path in paths:
        target = path if platform.system() == "Linux" else int(path)
        cap = cv2.VideoCapture(target)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()
            cameras.append({
                "device": path if platform.system() == "Linux" else int(path),
                "width": w,
                "height": h,
                "fps": round(fps, 1),
                "fourcc": fourcc,
            })
    return cameras


def _capture_snapshot_jpeg(device: str, quality: int = 80) -> bytes:
    """Open a camera, grab one frame, return JPEG bytes, then release."""
    # Check cache first
    now = time.time()
    cached = _snapshot_cache.get(device)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {device}")

    try:
        # Read a few frames to let auto-exposure settle
        for _ in range(3):
            ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {device}")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpeg_bytes = buf.tobytes()
        _snapshot_cache[device] = (now, jpeg_bytes)
        return jpeg_bytes
    finally:
        cap.release()


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
        # Fallback: scan /dev/ttyACM* and /dev/ttyUSB*
        for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
            for dev in sorted(glob.glob(pattern)):
                ports.append({"device": dev, "description": "", "manufacturer": "", "hwid": ""})
    return ports


def _read_current_config() -> dict[str, Any]:
    """Parse the current config.py and return ROBOT_CONFIG values."""
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text()
    # Extract the cameras string
    m = re.search(r'"cameras"\s*:\s*"([^"]*)"', content)
    cameras_str = m.group(1) if m else ""
    # Extract robot_port
    m2 = re.search(r'"robot_port"\s*:\s*"([^"]*)"', content)
    robot_port = m2.group(1) if m2 else ""
    # Parse cameras into dict
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
    """Update ROBOT_CONFIG in config.py with new camera mapping and optionally robot port."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    content = CONFIG_PATH.read_text()

    # Replace cameras value
    content = re.sub(
        r'("cameras"\s*:\s*)"([^"]*)"',
        f'\\1"{cameras_str}"',
        content,
    )

    # Optionally replace robot_port
    if robot_port:
        content = re.sub(
            r'("robot_port"\s*:\s*)"([^"]*)"',
            f'\\1"{robot_port}"',
            content,
        )

    CONFIG_PATH.write_text(content)


# ── API Models ─────────────────────────────────────────────────────────────

class CameraAssignment(BaseModel):
    role: str  # "front" or "wrist"
    device: str  # e.g. "/dev/video4"


class SaveConfigRequest(BaseModel):
    cameras: list[CameraAssignment]
    robot_port: str | None = None


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/preflight/detect-cameras")
async def detect_cameras():
    """Detect all available video devices and return their metadata."""
    loop = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, _detect_video_devices)
    return {"cameras": cameras}


@app.get("/api/preflight/snapshot/{device_path:path}")
async def get_snapshot(device_path: str):
    """
    Capture a snapshot from a specific video device.
    device_path should be like: dev/video4 (without leading /)
    The leading / is added back.
    """
    device = f"/{device_path}"
    if not os.path.exists(device):
        raise HTTPException(status_code=404, detail=f"Device {device} not found")

    loop = asyncio.get_event_loop()
    try:
        jpeg_bytes = await loop.run_in_executor(None, _capture_snapshot_jpeg, device)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/preflight/detect-ports")
async def detect_ports():
    """Detect available serial ports for robot connection."""
    loop = asyncio.get_event_loop()
    ports = await loop.run_in_executor(None, _detect_serial_ports)
    return {"ports": ports}


@app.get("/api/preflight/current-config")
async def get_current_config():
    """Return the current camera and port configuration."""
    config = _read_current_config()
    return {"config": config}


@app.post("/api/preflight/save-config")
async def save_config(req: SaveConfigRequest):
    """
    Save camera assignments and robot port to config.py.
    
    Expects:
      cameras: [{"role": "front", "device": "/dev/video4"}, ...]
      robot_port: "/dev/ttyACM0" (optional)
    """
    # Build cameras string: "front:/dev/video4,wrist:/dev/video6"
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


@app.get("/api/preflight/health")
async def health():
    return {"status": "ok", "service": "preflight"}
