#!/bin/bash
set -euo pipefail

# Ensure we're running from the script directory
cd "$(dirname "$0")" || exit 1

# Detect python binary (prefer python3)
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo
    echo "Stopping services..."
    # Stop frontend if running
    if [[ -n "${FRONTEND_PID-}" ]]; then
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            echo "Killing frontend (pid $FRONTEND_PID)..."
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
    fi
    # Stop backend if running
    if [[ -n "${BACKEND_PID-}" ]]; then
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo "Killing backend (pid $BACKEND_PID)..."
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
    fi
    # Wait for children to exit (best-effort)
    wait 2>/dev/null || true
}

#trap cleanup EXIT INT TERM

# Start backend
echo "Starting FastAPI backend..."
if [[ -z "$PYTHON_CMD" ]]; then
    echo "Warning: No python binary found (tried python3 and python). Skipping backend start."
else
    if [[ -d "backend" ]]; then
        cd backend
        # Install requirements (best-effort)
        echo "Installing Python requirements..."
        # allow pip install to fail without aborting the script
        ( "$PYTHON_CMD" -m pip install -r requirements.txt ) || echo "pip install failed or already satisfied"

        # Choose start method: denoise.py if present, otherwise uvicorn module
        if [[ -f "denoise.py" ]]; then
            echo "Launching backend via denoise.py..."
            nohup "$PYTHON_CMD" denoise.py > ../backend.log 2>&1 &
            BACKEND_PID=$!
        else
            echo "denoise.py not found, attempting to start uvicorn denoise:app..."
            nohup "$PYTHON_CMD" -m uvicorn denoise:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
            BACKEND_PID=$!
        fi

        # Poll for readiness (health endpoint or root)
        echo -n "Waiting for backend to respond"
        MAX_RETRIES=15
        COUNT=0
        while ! curl --silent --fail http://localhost:8000/health >/dev/null 2>&1 && ! curl --silent --fail http://localhost:8000/ >/dev/null 2>&1 && [ $COUNT -lt $MAX_RETRIES ]; do
            printf "."
            sleep 1
            COUNT=$((COUNT+1))
        done
        if curl --silent --fail http://localhost:8000/health >/dev/null 2>&1 || curl --silent --fail http://localhost:8000/ >/dev/null 2>&1; then
            echo " backend is up (pid $BACKEND_PID). Logs: backend.log"
        else
            echo " backend did not respond after $MAX_RETRIES seconds. Check backend.log for details."
        fi
        cd ..
    else
        echo "backend directory not found; skipping backend."
    fi
fi

# Start frontend
echo "Starting Next.js frontend..."
if [[ -d "frontend" ]]; then
    cd frontend

    echo "Installing npm dependencies..."
    # allow npm install to fail without aborting the script
    ( npm install ) || echo "npm install failed or already satisfied"

    # Detect available script: dev preferred, then start
    HAS_DEV="no"
    HAS_START="no"
    if [[ -f package.json ]]; then
        # Use grep to check for "dev" or "start" script presence
        if grep -q '"dev"\s*:' package.json >/dev/null 2>&1; then
            HAS_DEV="yes"
        fi
        if grep -q '"start"\s*:' package.json >/dev/null 2>&1; then
            HAS_START="yes"
        fi
    fi

    if [[ "$HAS_DEV" == "yes" ]]; then
        echo "Running 'npm run dev'..."
        nohup npm run dev > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
    elif [[ "$HAS_START" == "yes" ]]; then
        echo "No 'dev' script found; running 'npm start'..."
        nohup npm start > ../frontend.log 2>&1 &
        FRONTEND_PID=$!
    else
        echo "No 'dev' or 'start' script found in frontend/package.json. Skipping frontend start."
    fi

    cd ..
else
    echo "frontend directory not found; skipping frontend."
fi

echo "Backend running on http://localhost:8000 (log: backend.log)"
echo "Frontend running on http://localhost:3000 (log: frontend.log)"
echo "Press Ctrl+C to stop both services"

# Wait for background processes; trap will handle