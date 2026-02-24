#!/usr/bin/env bash
###############################################################################
#  One-click launcher for LeRobot Demo UI (backend + frontend)
#  Usage:  bash start.sh              # start both (main control mode)
#          bash start.sh preflight    # start in preflight camera setup mode
#          bash start.sh stop         # stop both
###############################################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/ui"

BACKEND_PORT=8000
FRONTEND_PORT=5173

CONDA_ENV="lerobot"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

PID_DIR="/tmp/lerobot_ui_pids"
mkdir -p "$PID_DIR"

# ─── helpers ────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;36m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR]\033[0m   $*"; }

kill_by_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        warn "Killing existing processes on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

stop_all() {
    info "Stopping LeRobot Demo UI..."

    # Kill by saved PIDs
    for f in "$PID_DIR"/backend.pid "$PID_DIR"/frontend.pid; do
        if [ -f "$f" ]; then
            local pid
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
                warn "Stopped PID $pid ($(basename "$f" .pid))"
            fi
            rm -f "$f"
        fi
    done

    # Also kill by port in case PIDs are stale
    kill_by_port $BACKEND_PORT
    kill_by_port $FRONTEND_PORT

    ok "All processes stopped."
}

# ─── stop mode ──────────────────────────────────────────────────────────────
if [ "${1:-}" = "stop" ]; then
    stop_all
    exit 0
fi

# ─── pre-flight checks ─────────────────────────────────────────────────────
if [ ! -f "$CONDA_SH" ]; then
    err "Conda not found at $CONDA_SH"
    exit 1
fi
if [ ! -d "$BACKEND_DIR" ]; then
    err "Backend directory not found: $BACKEND_DIR"
    exit 1
fi
if [ ! -d "$FRONTEND_DIR" ]; then
    err "Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

# ─── determine mode ─────────────────────────────────────────────────────────
MODE="control"
if [ "${1:-}" = "preflight" ]; then
    MODE="preflight"
fi

# ─── cleanup old processes ──────────────────────────────────────────────────
stop_all 2>/dev/null || true

# ─── start backend ──────────────────────────────────────────────────────────
if [ "$MODE" = "preflight" ]; then
    BACKEND_MODULE="preflight_server"
    info "Starting PREFLIGHT backend on port $BACKEND_PORT ..."
else
    BACKEND_MODULE="main_robot"
    info "Starting backend on port $BACKEND_PORT ..."
fi
(
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    cd "$BACKEND_DIR"
    exec uvicorn "$BACKEND_MODULE":app --host 0.0.0.0 --port "$BACKEND_PORT"
) &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$PID_DIR/backend.pid"
ok "Backend started  (PID: $BACKEND_PID, module: $BACKEND_MODULE)"

# ─── start frontend ────────────────────────────────────────────────────────
info "Starting frontend on port $FRONTEND_PORT ..."
(
    cd "$FRONTEND_DIR"
    exec npx vite --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$PID_DIR/frontend.pid"
ok "Frontend started (PID: $FRONTEND_PID)"

# ─── summary ───────────────────────────────────────────────────────────────
echo ""
if [ "$MODE" = "preflight" ]; then
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       🔧  LeRobot Preflight Check  — Running            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Frontend :  http://localhost:$FRONTEND_PORT                  ║"
echo "║  Backend  :  http://localhost:$BACKEND_PORT  (preflight)      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Stop all :  bash $(basename "$0") stop                      ║"
echo "║  Or press :  Ctrl+C                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
else
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           🤖  LeRobot Demo UI  — Running                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Frontend :  http://localhost:$FRONTEND_PORT                  ║"
echo "║  Backend  :  http://localhost:$BACKEND_PORT                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Stop all :  bash $(basename "$0") stop                      ║"
echo "║  Or press :  Ctrl+C                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
fi
echo ""

# ─── wait & handle Ctrl+C ──────────────────────────────────────────────────
trap 'echo ""; warn "Caught Ctrl+C — shutting down..."; stop_all; exit 0' INT TERM

wait
