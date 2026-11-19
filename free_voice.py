"""Free, entirely local Kokoro-ONNX voice for ChoiceForge.

No speech API, account, authentication, tracking or paid services.
One-time official model download is optional via install_models().
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
import wave

import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'models'
MODEL = MODEL_DIR / 'kokoro-v1.0.int8.onnx'
VOICES = MODEL_DIR / 'voices-v1.0.bin'
RELEASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/'
# Hashes published alongside the upstream release. Model downloads never execute code.
FILES = {
    'kokoro-v1.0.int8.onnx': 'ae315a79b623f244700e4afb9246c46a26066782e049ba174bf3ba433970ee9c',
    'voices-v1.0.bin': 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d',
}
VOICES_FREE = {
    'US · Heart (woman)': 'af_heart',
    'US · Bella (woman)': 'af_bella',
    'US · Sarah (woman)': 'af_sarah',
    'US · Nicole (woman)': 'af_nicole',
    'US · Adam (man)': 'am_adam',
    'US · Michael (man)': 'am_michael',
    'UK · Emma (woman)': 'bf_emma',
    'UK · Isabella (woman)': 'bf_isabella',
    'UK · George (man)': 'bm_george',
    'UK · Lewis (man)': 'bm_lewis',
}


@lru_cache(maxsize=8)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    s = path.stat()
    return _sha256_cached(str(path), s.st_size, s.st_mtime_ns)


def model_status() -> dict[str, bool]:
    return {name: (MODEL_DIR / name).is_file() and sha256_file(MODEL_DIR / name) == digest
            for name, digest in FILES.items()}


def install_models(progress=None) -> list[str]:
    """Download official models, verify checksums, then atomically commit them."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    for filename, digest in FILES.items():
        path = MODEL_DIR / filename
        if path.exists() and sha256_file(path) == digest:
            continue
        # Never replace an existing file until the new download verifies.
        fd, name = tempfile.mkstemp(dir=MODEL_DIR, prefix='.download-', suffix='.part')
        os.close(fd)
        temp = Path(name)
        try:
            req = urllib.request.Request(RELEASE + filename, headers={'User-Agent': 'ChoiceForge-Free/4.0'})
            with urllib.request.urlopen(req, timeout=40) as response, temp.open('wb') as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    if progress:
                        progress(filename, temp.stat().st_size)
            if sha256_file(temp) != digest:
                raise RuntimeError(f'Checksum mismatch for {filename}; refusing to install a corrupt or replaced model.')
            os.replace(temp, path)
            fetched.append(filename)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError('Could not download free speech models from official GitHub. '
                               'Check your connection, or download them manually from: '
                               + RELEASE + '\n' + str(exc)) from exc
        finally:
            temp.unlink(missing_ok=True)
    return fetched


@lru_cache(maxsize=1)
def _load_model(model_file: str, voices_file: str):
    from kokoro_onnx import Kokoro
    return Kokoro(model_file, voices_file)


def synthesize(text: str, output: Path, voice: str = 'af_heart', speed: float = 1.0) -> Path:
    """Produce 24kHz WAV with Kokoro ONNX on CPU. No robotic fallback."""
    if not text.strip():
        raise ValueError('Speech text cannot be empty')
    if voice not in VOICES_FREE.values():
        raise ValueError('Unsupported Kokoro voice: ' + voice)
    if not 0.7 <= speed <= 1.3:
        raise ValueError('Kokoro speech speed must be between 0.70 and 1.30')
    if not all(model_status().values()):
        raise RuntimeError('Free Kokoro model is not installed or failed checksum verification. '
                           'Use the app button "DOWNLOAD FREE VOICE MODEL" or run: python setup_free_voice.py')
    try:
        model = _load_model(str(MODEL), str(VOICES))
    except ImportError as exc:
        raise RuntimeError('The free Kokoro runtime is not installed. Run: pip install -r requirements.txt') from exc
    lang = 'en-gb' if voice.startswith(('bf_', 'bm_')) else 'en-us'
    try:
        audio, sample_rate = model.create(text, voice=voice, speed=float(speed), lang=lang)
    except Exception as exc:
        raise RuntimeError(f'Local neural narration failed: {exc}') from exc
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(arr) == 0 or not np.all(np.isfinite(arr)):
        raise RuntimeError('Local neural model returned invalid audio')
    if not 16000 <= int(sample_rate) <= 48000:
        raise RuntimeError('Local neural model returned an unexpected sample rate')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes((np.clip(arr, -0.96, 0.96) * 32767).astype('<i2').tobytes())
    return output
