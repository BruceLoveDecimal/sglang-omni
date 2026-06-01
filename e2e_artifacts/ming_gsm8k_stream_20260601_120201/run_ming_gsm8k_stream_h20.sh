#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/autodl-tmp/sglang-omni-h20}
CODE=$BASE/code/sglang-omni
VENV=$BASE/venvs/sglang-omni-py312
MODEL_PATH=$BASE/models/Ming-flash-omni-2.0
PORT=${PORT:-8000}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-60}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
LIMIT=${LIMIT:-32}
CONCURRENCY=${CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-256}
WARMUP=${WARMUP:-1}
DATASET_JSONL=${DATASET_JSONL:-$BASE/data/gsm8k_test.jsonl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HF_HOME=$BASE/hf-cache
export HF_HUB_CACHE=$BASE/hf-cache/hub
export PYTHONPATH=$CODE:${PYTHONPATH:-}

source "$VENV/bin/activate"
cd "$CODE"
mkdir -p "$BASE/logs" "$BASE/results"

STAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=$BASE/results/ming_gsm8k_stream_$STAMP
SERVER_LOG=$BASE/logs/ming_gsm8k_server_$STAMP.log
BENCH_LOG=$BASE/logs/ming_gsm8k_benchmark_$STAMP.log
SERVER_PID=

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server process group $SERVER_PID"
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    sleep 8
    kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

setsid bash -lc "
source '$VENV/bin/activate'
cd '$CODE'
export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
export OMP_NUM_THREADS='$OMP_NUM_THREADS'
export HF_HOME='$HF_HOME'
export HF_HUB_CACHE='$HF_HUB_CACHE'
export PYTHONPATH='$PYTHONPATH'
exec python examples/run_ming_omni_server.py \
  --model-path '$MODEL_PATH' \
  --port '$PORT' \
  --tp-size 2 \
  --cpu-offload-gb '$CPU_OFFLOAD_GB' \
  --mem-fraction-static '$MEM_FRACTION_STATIC' \
  --relay-backend shm
" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "server_pid=$SERVER_PID"
echo "server_log=$SERVER_LOG"
echo "result_dir=$RESULT_DIR"
echo "benchmark_log=$BENCH_LOG"

python - <<PY
import requests
import sys
import time

url = "http://127.0.0.1:$PORT/v1/models"
for i in range(240):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print("server_ready", i, r.text[:200])
            sys.exit(0)
    except Exception as e:
        if i % 10 == 0:
            print("waiting_for_server", i, repr(e))
    time.sleep(10)
print("server_not_ready")
sys.exit(1)
PY

python "$BASE/benchmark_ming_gsm8k_stream.py" \
  --url "http://127.0.0.1:$PORT" \
  --model ming-omni \
  --split test \
  --dataset-jsonl "$DATASET_JSONL" \
  --limit "$LIMIT" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --warmup "$WARMUP" \
  --output-dir "$RESULT_DIR" \
  2>&1 | tee "$BENCH_LOG"

cp "$SERVER_LOG" "$RESULT_DIR/server.log"
cp "$BENCH_LOG" "$RESULT_DIR/benchmark.log"
echo "$RESULT_DIR"
