#!/usr/bin/env python3
"""
Camera Worker — runs in a SEPARATE subprocess from the FastAPI server.

This script is invoked by preflight_server.py via subprocess.run().
It handles all OpenCV camera operations in isolation so that:
  1. The FastAPI process never loads OpenCV / CUDA
  2. No CUDA driver corruption can occur
  3. Short-lived subprocess cleans up all GPU/camera resources on exit

Usage:
  python _camera_worker.py detect            → JSON list of cameras to stdout
  python _camera_worker.py snapshot <device>  → JPEG bytes to stdout
"""

import glob
import json
import os
import platform
import re
import sys

# ── Force CPU-only — prevent any CUDA context in this worker ──────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

_CV2_BACKEND = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
WARMUP_FRAMES = 5
SOLID_COLOR_THRESHOLD = 0.97


def is_capture_device(path: str) -> bool:
    """Even-numbered /dev/video* = capture stream; odd = metadata (skip)."""
    try:
        num = int(re.search(r"\d+$", path).group())
        return num % 2 == 0
    except (AttributeError, ValueError):
        return True


def is_solid_color_frame(frame) -> bool:
    """Return True if frame is dominated by a single solid color (unusable)."""
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(gray))
    within = np.sum(np.abs(gray.astype(float) - median_val) < 12)
    fraction = within / gray.size
    return fraction > SOLID_COLOR_THRESHOLD


def detect_cameras() -> list[dict]:
    """Scan /dev/video* and return metadata for each usable camera."""
    cameras = []
    if platform.system() == "Linux":
        all_paths = sorted(glob.glob("/dev/video*"))
        paths = [p for p in all_paths if is_capture_device(p)]
    else:
        paths = [str(i) for i in range(20)]

    for path in paths:
        target = path if platform.system() == "Linux" else int(path)
        cap = cv2.VideoCapture(target, _CV2_BACKEND)
        if not cap.isOpened():
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        last_frame = None
        for _ in range(WARMUP_FRAMES + 1):
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frame = frame
        cap.release()

        if last_frame is not None and not is_solid_color_frame(last_frame):
            cameras.append({
                "device": path if platform.system() == "Linux" else int(path),
                "width": w,
                "height": h,
                "fps": round(fps, 1),
                "fourcc": fourcc,
            })

    return cameras


def capture_snapshot(device: str, quality: int = 80) -> bytes:
    """Open camera, grab one stable frame, return JPEG bytes."""
    cap = cv2.VideoCapture(device, _CV2_BACKEND)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {device}")

    try:
        ret, frame = False, None
        for _ in range(WARMUP_FRAMES + 1):
            ret, frame = cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {device}")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
    finally:
        cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _camera_worker.py detect | snapshot <device>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "detect":
        cameras = detect_cameras()
        # Write JSON to stdout
        json.dump(cameras, sys.stdout)
        sys.stdout.flush()

    elif command == "snapshot":
        if len(sys.argv) < 3:
            print("Usage: _camera_worker.py snapshot <device>", file=sys.stderr)
            sys.exit(1)
        device = sys.argv[2]
        try:
            jpeg_bytes = capture_snapshot(device)
            # Write raw JPEG bytes to stdout (binary mode)
            sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)
