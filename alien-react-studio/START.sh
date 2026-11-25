#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m http.server 8765 --bind 127.0.0.1 &
pid=$!
trap 'kill "$pid" 2>/dev/null' EXIT
sleep 1
xdg-open http://localhost:8765
wait "$pid"
