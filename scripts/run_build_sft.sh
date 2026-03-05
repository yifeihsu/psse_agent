#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip -q install fastmcp requests numpy pandas tqdm uvicorn starlette >/dev/null 2>&1 || true

HOST="127.0.0.1"
PORT="3929"
PATHPFX="/tools"

python -m mcp_server.run_http_server --host "$HOST" --port "$PORT" --path "$PATHPFX" \
  > mcp_server/matpower_server.out.log 2> mcp_server/matpower_server.err.log &
SVR_PID=$!

# Wait up to 60s for server to listen
for i in $(seq 1 60); do
  if ss -ltn | awk '{print $4}' | grep -q ":$PORT$"; then echo "Server listening on $PORT in $i s"; break; fi
  sleep 1
  if ! ps -p $SVR_PID > /dev/null 2>&1; then echo "Server exited early"; break; fi
done

python Transmission/build_sft_traces.py \
  --samples out_sft_measurements/samples.jsonl \
  --meta out_sft_measurements/meta.json \
  --endpoint "http://$HOST:$PORT$PATHPFX" \
  --out sft_with_tools.jsonl || true

LINES=$(wc -l < sft_with_tools.jsonl || echo 0)

if ps -p $SVR_PID > /dev/null 2>&1; then kill $SVR_PID || true; fi

if [ "${LINES:-0}" -eq 0 ]; then
  echo "No SFT rows produced with real tool; running mock mode." >&2
  python Transmission/build_sft_traces.py \
    --samples out_sft_measurements/samples.jsonl \
    --meta out_sft_measurements/meta.json \
    --endpoint "http://$HOST:$PORT$PATHPFX" \
    --out sft_with_tools.jsonl \
    --mock
fi

wc -l sft_with_tools.jsonl || true
ls -la sft_with_tools.jsonl split_*.jsonl || true

