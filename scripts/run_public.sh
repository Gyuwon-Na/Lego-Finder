#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
CLOUDFLARED_BIN="$ROOT_DIR/.tools/cloudflared"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing virtualenv. Run: python3 -m venv .venv"
  exit 1
fi

if [ ! -x "$CLOUDFLARED_BIN" ]; then
  echo "Missing cloudflared. Expected: $CLOUDFLARED_BIN"
  exit 1
fi

cd "$ROOT_DIR"

echo "Starting BrickFinder on http://127.0.0.1:$PORT"
SERVER_PID=""
if "$PYTHON_BIN" -c "import socket; s=socket.socket(); s.settimeout(0.5); raise SystemExit(s.connect_ex(('127.0.0.1', int('$PORT'))))"; then
  echo "Port $PORT is already serving; using the existing server."
else
  "$PYTHON_BIN" -m uvicorn backend.app.main:app --host 127.0.0.1 --port "$PORT" &
  SERVER_PID=$!
fi

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

sleep 2

echo
echo "Opening public HTTPS tunnel. Use the https://*.trycloudflare.com URL on your phone."
"$CLOUDFLARED_BIN" tunnel --protocol http2 --url "http://127.0.0.1:$PORT"
