#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: FFmpeg and ffprobe must be installed first."
  echo "macOS: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg"
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
.venv/bin/python -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Python 3.11–3.13 required; remove .venv and reinstall with Python 3.12"'
if [[ ! -f .venv/.installed-v4 ]]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed-v4
fi
.venv/bin/python app.py
