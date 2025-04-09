#!/bin/bash

set -e

# Log file path
LOG_FILE="memory/launch_log.txt"

# Detect if inside *any* container (dev container or Docker)
if grep -qaE 'docker|containerd|/docker/' /proc/1/cgroup 2>/dev/null || [ -f /.dockerenv ] || [ "$ALT_IN_CONTAINER" = "1" ]; then
    echo "[INFO] Detected container environment. Running AltLAS UI directly..."
    python altlas_ui.py "$@"
    exit $?
fi

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or not in PATH. Please install Docker."
    exit 127
fi

# Log launch timestamp
mkdir -p memory
if [ -w memory ]; then
    echo "$(date -u +'%Y-%m-%d %H:%M:%S UTC') : Launching AltLAS with args: $*" >> "$LOG_FILE"
fi

# Run Docker container
exec docker run --rm -it \
    -v "$(pwd)":/app \
    -w /app \
    -e ALT_IN_CONTAINER=1 \
    altlas-image:latest \
    python altlas_ui.py "$@"
