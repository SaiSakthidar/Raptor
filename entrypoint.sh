#!/bin/bash
set -e

# If red output is missing, generate attack data first
if [ ! -f red/output/V001_structuring.jsonl ]; then
    echo "=== Generating attack data ==="
    python -m red.run_all
fi

# If model artifacts are missing, run the full blue stack
if [ ! -f blue/results/txn_sequence_model.joblib ]; then
    echo "=== Training detection models ==="
    python -m blue.run_blue
fi

echo "=== Starting RAPTOR dashboard on http://0.0.0.0:8080 ==="
exec uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 ${RELOAD:+--reload}
