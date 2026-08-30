#!/bin/bash
# AEGIS — run helper
# Usage:
#   bash run.sh                   → run blue stack (train + evaluate)
#   bash run.sh --red             → generate attack data (all 21 vectors)
#   bash run.sh --ui              → start dashboard on http://localhost:8080
#   bash run.sh --all             → red → blue → ui (full pipeline)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="mastercard-aegis"

function run_docker() {
  docker run --rm -v "$SCRIPT_DIR:/app" "$IMAGE" "$@"
}

function run_ui() {
  echo "Starting AEGIS dashboard → http://localhost:8080"
  docker run --rm \
    -v "$SCRIPT_DIR:/app" \
    -p 8080:8080 \
    "$IMAGE" \
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
}

case "${1:-}" in
  --red)
    echo "=== Generating attack data (21 vectors) ==="
    run_docker python -m red.run_all
    ;;
  --ui)
    run_ui
    ;;
  --all)
    echo "=== Step 1/3: Generating attack data ==="
    run_docker python -m red.run_all
    echo "=== Step 2/3: Training + evaluating blue stack ==="
    run_docker python -m blue.run_blue
    echo "=== Step 3/3: Starting dashboard ==="
    run_ui
    ;;
  *)
    echo "=== Running blue stack ==="
    run_docker python -m blue.run_blue
    ;;
esac
