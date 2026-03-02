#!/usr/bin/env python3
"""
Camera Worker — runs in a SEPARATE subprocess from the FastAPI server.

This script is invoked by preflight_server.py and supports three modes:

  1. detect   — scan /dev/video* and print JSON camera list to stdout (one-shot)
  2. snapshot — open camera, grab one frame, write JPEG to stdout (one-shot)
  3. serve    — keep cameras open, write latest JPEG frames to a shared directory
                so the preflight server can serve them without re-opening cameras.

Mode 3 ("serve") solves the green-frame / instability problem on YUYV cameras
(e.g. RealSense RGB on /dev/video5) that occurs when cameras are opened and
closed repeatedly for every snapshot request.

Usage:
  python _camera_worker.py detect
  python _camera_worker.py snapshot <device>
  python _camera_worker.py serve <frame_dir> <dev1> [<dev2> ...]
"""

import glob
import json
import os
import platform
import re
import signal
import sys
import time

# ── Force CPU-only — prevent any CUDA context in this worker ──────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

# NOTE: We use CAP_ANY (not CAP_V4L2) — V4L2 backend can cause green tint
# on some cameras due to incorrect YUYV→BGR conversion.
# CUDA safety is already ensured by CUDA_VISIBLE_DEVICES="" above.
WARMUP_FRAMES = 2    # fewer warmup frames = faster snapshots
SOLID_COLOR_THRESHOLD = 0.95  # reject frames that are >95% one color (green/black)


def _is_capture_device(path: str) -> bool:
    """Check if a /dev/video* node is a VIDEO_CAPTURE device (not metadata-only).

    Uses V4L2 capability flags when available; falls back to allowing all devices.
    The old even/odd heuristic was wrong — many cameras expose usable streams
    on odd-numbered device nodes (e.g. /dev/video5, /dev/video7).
    """
    try:
        import fcntl
        import ctypes

        # V4L2 VIDIOC_QUERYCAP ioctl
        VIDIOC_QUERYCAP = 0x80685600

        class v4l2_capability(ctypes.Structure):
            _fields_ = [
                ("driver", ctypes.c_char * 16),
                ("card", ctypes.c_char * 32),
                ("bus_info", ctypes.c_char * 32),
                ("version", ctypes.c_uint32),
                ("capabilities", ctypes.c_uint32),
                ("device_caps", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 3),
            ]

        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            cap = v4l2_capability()
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
            # Use device_caps if V4L2_CAP_DEVICE_CAPS (0x80000000) is set
            caps = cap.device_caps if (cap.capabilities & 0x80000000) else cap.capabilities
            V4L2_CAP_VIDEO_CAPTURE = 0x00000001
            return bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        finally:
            os.close(fd)
    except Exception:
        # If ioctl fails, allow the device — OpenCV's isOpened() will filter later
        return True


def _set_mjpeg(cap) -> bool:
    """Try to switch camera to MJPEG format — faster USB transfer, no color issues.

    Returns True if MJPEG was successfully set, False otherwise.
    Some cameras (e.g. RealSense RGB on /dev/video5) only support YUYV;
    forcing MJPEG on them causes green frames and instability.
    """
    mjpg = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    # Verify the camera actually accepted MJPEG
    actual = int(cap.get(cv2.CAP_PROP_FOURCC))
    if actual == mjpg:
        return True
    # Camera rejected MJPEG — leave it on its native format (e.g. YUYV)
    return False


def detect_cameras() -> list[dict]:
    """
    Scan /dev/video* and return metadata for each openable camera.
    Fast: only checks isOpened() + reads metadata. No frame capture.
    """
    cameras = []
    if platform.system() == "Linux":
        all_paths = sorted(glob.glob("/dev/video*"))
        paths = [p for p in all_paths if _is_capture_device(p)]
    else:
        paths = [str(i) for i in range(20)]

    for path in paths:
        target = path if platform.system() == "Linux" else int(path)
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            continue

        _set_mjpeg(cap)  # best-effort; camera keeps native format if MJPEG unsupported
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
            "fourcc": fourcc.strip('\x00'),
        })

    return cameras


def _is_solid_color(frame) -> bool:
    """Return True if >95% of pixels are within ±12 of the median (solid green/black)."""
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(gray))
    within = np.sum(np.abs(gray.astype(float) - median_val) < 12)
    return (within / gray.size) > SOLID_COLOR_THRESHOLD


def capture_snapshot(device: str, quality: int = 80) -> bytes:
    """Open camera, grab one stable frame, return JPEG bytes.
    Rejects solid-color frames (green/black from metadata or depth nodes).

    For YUYV-only cameras (e.g. RealSense RGB) we use more warmup frames
    and a retry loop because the first few frames are often green/corrupt.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {device}")

    try:
        is_mjpeg = _set_mjpeg(cap)
        # YUYV cameras (RealSense) need more warmup frames to stabilise
        warmup = WARMUP_FRAMES if is_mjpeg else WARMUP_FRAMES + 6

        ret, frame = False, None
        for _ in range(warmup + 1):
            ret, frame = cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {device}")

        # For YUYV cameras, retry a few times if we get a solid-color (green) frame
        if not is_mjpeg and _is_solid_color(frame):
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and not _is_solid_color(frame):
                    break

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {device}")

        if _is_solid_color(frame):
            raise RuntimeError(f"Device {device} returned a solid-color frame (unusable)")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
    finally:
        cap.release()


# ═══════════════════════════════════════════════════════════════════════════
#  SERVE MODE — persistent camera capture loop
# ═══════════════════════════════════════════════════════════════════════════

def serve_cameras(frame_dir: str, devices: list[str],
                  fps: float = 2.0, quality: int = 80) -> None:
    """Keep cameras open and continuously write latest JPEG frames to disk.

    For each device /dev/videoN, writes:
        <frame_dir>/videoN.jpg      (latest good frame)
        <frame_dir>/videoN.meta     (JSON: timestamp, status, fourcc)

    The preflight server reads these files instead of spawning subprocesses.
    This avoids the open/close churn that causes green frames on YUYV cameras.
    """
    os.makedirs(frame_dir, exist_ok=True)
    interval = 1.0 / fps

    # ── Open all cameras ──
    captures: dict[str, cv2.VideoCapture] = {}
    cam_info: dict[str, dict] = {}  # device → {is_mjpeg, warmup_done, fail_count}

    for dev in devices:
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print(f"[serve] WARN: Cannot open {dev}, skipping", file=sys.stderr)
            cap.release()
            # Write error meta so preflight server knows
            _write_meta(frame_dir, dev, "error", "Cannot open device")
            continue

        is_mjpeg = _set_mjpeg(cap)
        captures[dev] = cap
        cam_info[dev] = {"is_mjpeg": is_mjpeg, "warmup_done": False, "fail_count": 0}
        print(f"[serve] Opened {dev} (format={'MJPEG' if is_mjpeg else 'YUYV'})",
              file=sys.stderr)

    if not captures:
        print("[serve] ERROR: No cameras could be opened", file=sys.stderr)
        sys.exit(1)

    # ── Warmup: read several frames to let cameras stabilise ──
    print("[serve] Warming up cameras...", file=sys.stderr)
    for dev, cap in captures.items():
        info = cam_info[dev]
        warmup_count = 3 if info["is_mjpeg"] else 15  # YUYV needs more warmup
        for _ in range(warmup_count):
            cap.read()
        info["warmup_done"] = True
    print("[serve] Warmup complete, starting capture loop", file=sys.stderr)

    # ── Graceful shutdown ──
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # ── Main capture loop ──
    try:
        while running:
            t0 = time.monotonic()

            for dev, cap in list(captures.items()):
                info = cam_info[dev]
                ret, frame = cap.read()

                if not ret or frame is None:
                    info["fail_count"] += 1
                    if info["fail_count"] > 10:
                        # Too many failures — try to reopen
                        print(f"[serve] Reopening {dev} after {info['fail_count']} failures",
                              file=sys.stderr)
                        cap.release()
                        new_cap = cv2.VideoCapture(dev)
                        if new_cap.isOpened():
                            info["is_mjpeg"] = _set_mjpeg(new_cap)
                            captures[dev] = new_cap
                            info["fail_count"] = 0
                            # Quick warmup
                            for _ in range(5):
                                new_cap.read()
                        else:
                            new_cap.release()
                            _write_meta(frame_dir, dev, "error", "Device lost")
                    continue

                info["fail_count"] = 0

                # Skip solid-color (green) frames — don't overwrite last good frame
                if _is_solid_color(frame):
                    continue

                # Encode and write atomically (write to .tmp then rename)
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, quality])
                jpg_bytes = buf.tobytes()

                fname = _device_to_filename(dev)
                tmp_path = os.path.join(frame_dir, f"{fname}.tmp")
                final_path = os.path.join(frame_dir, f"{fname}.jpg")

                with open(tmp_path, "wb") as f:
                    f.write(jpg_bytes)
                os.replace(tmp_path, final_path)  # atomic on Linux

                _write_meta(frame_dir, dev, "ok", "",
                            fourcc="MJPEG" if info["is_mjpeg"] else "YUYV")

            # Sleep to maintain target FPS
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("[serve] Shutting down, releasing cameras...", file=sys.stderr)
        for cap in captures.values():
            cap.release()
        print("[serve] Done.", file=sys.stderr)


def _device_to_filename(device: str) -> str:
    """Convert /dev/video5 → 'video5'."""
    return device.replace("/dev/", "").replace("/", "_")


def _write_meta(frame_dir: str, device: str, status: str, message: str = "",
                fourcc: str = "") -> None:
    """Write a small JSON metadata file for a device."""
    fname = _device_to_filename(device)
    meta = {
        "device": device,
        "status": status,
        "message": message,
        "fourcc": fourcc,
        "timestamp": time.time(),
    }
    meta_path = os.path.join(frame_dir, f"{fname}.meta")
    tmp_path = os.path.join(frame_dir, f"{fname}.meta.tmp")
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_path, meta_path)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _camera_worker.py detect | snapshot <device> | serve <frame_dir> <dev1> [dev2 ...]",
              file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "detect":
        cameras = detect_cameras()
        json.dump(cameras, sys.stdout)
        sys.stdout.flush()

    elif command == "snapshot":
        if len(sys.argv) < 3:
            print("Usage: _camera_worker.py snapshot <device>", file=sys.stderr)
            sys.exit(1)
        device = sys.argv[2]
        try:
            jpeg_bytes = capture_snapshot(device)
            sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    elif command == "serve":
        if len(sys.argv) < 4:
            print("Usage: _camera_worker.py serve <frame_dir> <dev1> [dev2 ...]",
                  file=sys.stderr)
            sys.exit(1)
        frame_dir = sys.argv[2]
        devices = sys.argv[3:]
        serve_cameras(frame_dir, devices)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)
