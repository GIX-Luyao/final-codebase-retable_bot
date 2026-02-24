# LeRobot Pick & Place — System Runbook

Complete guide for launching the LeRobot pick-and-place demo system.

---

## Quick Start (TL;DR)

```bash
# 1. Activate environment
conda activate lerobot

# 2. Run preflight to configure cameras & robot port
bash ~/lerobot/launch-lerobot-demo-ui/start.sh preflight

# 3. Open browser → http://localhost:5173
#    Assign front/wrist cameras, select robot port, save config

# 4. Stop preflight, then launch the main control UI
bash ~/lerobot/launch-lerobot-demo-ui/start.sh stop
bash ~/lerobot/launch-lerobot-demo-ui/start.sh

# 5. Open browser → http://localhost:5173  → control the robot
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Conda env** | `lerobot` (Python 3.10+) |
| **GPU** | CUDA-capable GPU (model runs on `cuda`) |
| **Robot** | SO-101 / HOPE arm connected via USB serial (`/dev/ttyACM*`) |
| **Cameras** | 2 USB cameras — one for front workspace view, one for wrist |
| **Node.js** | Required for the Vite frontend dev server |

### Install backend dependencies (first time only)

```bash
conda activate lerobot
pip install fastapi uvicorn[standard] pydantic websockets opencv-python pyserial
```

### Install frontend dependencies (first time only)

```bash
cd ~/lerobot/launch-lerobot-demo-ui/ui
npm install
```

---

## Step-by-Step Launch Guide

### Step 1 — Activate the Conda Environment

```bash
conda activate lerobot
```

All subsequent commands assume this environment is active.

---

### Step 2 — Preflight Check: Camera & Port Configuration

**Why this step matters:** USB cameras get assigned different `/dev/video*` paths on each boot or re-plug. You need to visually verify which camera is the front camera and which is the wrist camera, then save that mapping.

#### Option A: Use the Preflight UI (Recommended)

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh preflight
```

This launches:
- **Backend** on `http://localhost:8000` — the preflight server that detects cameras and ports
- **Frontend** on `http://localhost:5173` — the preflight UI

Open **http://localhost:5173** in a browser. The UI will:

1. **Auto-detect** all connected cameras and serial ports
2. **Show live previews** from each camera so you can visually identify them
3. **Let you assign roles** — click "Front" or "Wrist" on each camera card
4. **Select the robot port** (e.g., `/dev/ttyACM0`)
5. **Save** — writes the mapping to `launch-lerobot-demo-ui/backend/config.py`

After saving, stop the preflight server:

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh stop
```

#### Option B: Manual Configuration

If you prefer manual setup:

1. **Find cameras:**
   ```bash
   lerobot-find-cameras opencv
   ```
   This lists all available video devices with their `/dev/video*` paths.

2. **Identify cameras visually:**
   ```bash
   python ~/lerobot/view_cameras_live.py
   ```
   Check which device path shows the front view vs wrist view.

3. **Find robot port:**
   ```bash
   lerobot-find-port
   ```
   Typically `/dev/ttyACM0` or `/dev/ttyACM1`.

4. **Edit config.py:**
   ```bash
   nano ~/lerobot/launch-lerobot-demo-ui/backend/config.py
   ```
   Update these values in `ROBOT_CONFIG`:
   ```python
   "robot_port": "/dev/ttyACM0",          # your robot port
   "cameras": "front:/dev/video4,wrist:/dev/video6",  # your camera mapping
   ```

---

### Step 3 — Launch the Main Control UI

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh
```

This launches:
- **Backend** (`main_robot`) on `http://localhost:8000` — spawns `eval_act_safe.py` with model pre-warming
- **Frontend** on `http://localhost:5173` — the robot control UI

Open **http://localhost:5173** in a browser.

---

### Step 4 — Wait for Model Warmup

The backend automatically:
1. Loads the ACT model (`FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202`)
2. Connects to the robot
3. Initializes cameras

The UI shows a **WARMUP** state with a progress bar. This takes ~30 seconds.

---

### Step 5 — Run Inference

Once the state changes to **READY**:

| Button | Action |
|---|---|
| **▶ Start Inference** | Begin the pick-and-place task |
| **⏸ Emergency Stop** | Immediate e-stop |
| **🏠 Home** | Move robot to home position |
| **▶ Resume** | Continue after pause |
| **✕ Quit** | Stop inference and re-warm |

The UI also shows:
- **Live camera feeds** (front + wrist)
- **Hand safety detection** — auto-stops if a hand enters the front camera view
- **Feedback modal** — rate task success after completion

---

## Stopping the System

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh stop
```

Or press **Ctrl+C** in the terminal where `start.sh` is running.

---

## Configuration Reference

All robot configuration is in `~/lerobot/launch-lerobot-demo-ui/backend/config.py`:

```python
ROBOT_CONFIG = {
    "model": "FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202",
    "robot_port": "/dev/ttyACM0",
    "robot_id": "hope",
    "cameras": "front:/dev/video4,wrist:/dev/video6",
    "fps": 30,
    "episode_time": 200,
    "num_episodes": 10,
    "device": "cuda",
    "rest_duration": 2.0,
}
```

| Parameter | Description |
|---|---|
| `model` | HuggingFace model ID for the ACT policy |
| `robot_port` | Serial device for the robot arm |
| `robot_id` | Robot identifier (e.g., `hope`) |
| `cameras` | Comma-separated `role:/dev/videoN` pairs |
| `fps` | Inference frame rate |
| `episode_time` | Max seconds per episode |
| `num_episodes` | Number of episodes to run |
| `device` | `cuda` or `cpu` |
| `rest_duration` | Seconds to pause between episodes |

### Hand Safety Settings

```python
HAND_DETECT_ENABLED = True        # Toggle hand detection
HAND_DETECT_CAMERA = "front"      # Camera to monitor
HAND_DETECT_INTERVAL = 0.25       # Check frequency (seconds)
HAND_DETECT_COOLDOWN = 8          # Frames before auto-resume
```

---

## Troubleshooting

### Cameras not detected
- Check USB connections: `ls /dev/video*`
- Try: `lerobot-find-cameras opencv`
- Some `/dev/video*` entries are metadata-only (even numbers are usually the actual streams)

### Robot port not found
- Check USB: `ls /dev/ttyACM*`
- Try: `lerobot-find-port`
- Unplug/replug the robot USB cable

### Frontend won't load
- Ensure npm dependencies installed: `cd ~/lerobot/launch-lerobot-demo-ui/ui && npm install`
- Check port 5173 isn't in use: `lsof -i :5173`

### Backend crash on startup
- Ensure conda env is active: `conda activate lerobot`
- Check Python deps: `pip install fastapi uvicorn opencv-python`
- Check port 8000 isn't in use: `lsof -i :8000`

### Model warmup fails
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check model exists: the model is auto-downloaded from HuggingFace on first run

---

## Architecture Overview

```
launch-lerobot-demo-ui/
├── start.sh                  # One-click launcher (supports: preflight | stop)
├── backend/
│   ├── config.py             # Robot configuration (cameras, port, model)
│   ├── preflight_server.py   # Preflight camera calibration API
│   ├── main_robot.py         # Main robot control API + WebSocket
│   └── main.py               # Mock backend (for testing without robot)
└── ui/src/
    ├── main.tsx              # Entry point — auto-detects preflight vs control mode
    ├── PreflightCheck.tsx    # Camera assignment wizard UI
    └── App.tsx               # Main robot control UI
```

### Ports Used

| Port | Service |
|---|---|
| `8000` | FastAPI backend (preflight or main_robot) |
| `5173` | Vite dev server (frontend) |

### Communication Flow

```
Browser (5173) → Vite proxy → FastAPI (8000) → eval_act_safe.py (subprocess)
                                      ↕ WebSocket (real-time state)
                                      ↕ Control file (/tmp/lerobot_cmd)
                                      ↕ Frame dir (/tmp/lerobot_frames)
```
