# ChoiceForge Studio v4 — 100% free local neural voices

A local video creator for original **Would You Rather / This or That / Pick One** challenges. Creates portrait Shorts or 16:9 videos with subtle motion, sound effects, captions, editable narration stems, thumbnails, and optional permitted background music.

**No subscription, no paid voice API, no API key.** Speech uses the free/open-weight [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) neural model via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). The model is Apache-2.0 and the ONNX Python runner is MIT. The program itself is under `LICENSE`.

**What is not included in the small ZIP:** roughly **136 MB** of free Kokoro model files. They are downloaded **once** from official GitHub releases when you click the app's model installer button or run `python setup_free_voice.py`. A network connection and available disk space are needed for installation. The installer checks upstream SHA-256 hashes, refuses corrupted files, and never requests payment. Once installed, video rendering and voice narration run locally without network access.

> The current package was tested on Linux using synthetic test speech and a simulated ONNX speech engine. We could not download or audition the genuine model in this test environment. Its subjective naturalness, its real speed on your Lenovo, and Windows/macOS compatibility have not been certified; don't assume perfection until you audition it yourself.

## 1. Install free prerequisites

**Python:** use 3.11, 3.12 or 3.13 (**3.14 is not supported by the pinned Kokoro runtime as of this build**). Install [Python](https://www.python.org/downloads/) and [FFmpeg](https://ffmpeg.org/download.html). Both are free.

**Windows**
1. Install Python 3.12, checking **Add Python to PATH**.
2. Install FFmpeg with PowerShell: `winget install --id Gyan.FFmpeg -e`.
3. Close and reopen your terminal, check `ffmpeg -version` and `ffprobe -version`.
4. Extract this ZIP, double-click `start_windows.bat`, and allow the *free* first-time Python package installation.
5. Open `http://127.0.0.1:7860` if the browser does not open automatically.

**macOS** (requires free Homebrew or preinstalled Python/FFmpeg):
```bash
brew install python@3.12 ffmpeg
cd ChoiceForge_Studio_v4_Free
python3.12 -m venv .venv
./start_linux_mac.sh
```

**Ubuntu / Debian Linux:**
```bash
sudo apt update && sudo apt install -y ffmpeg python3 python3-venv
cd ChoiceForge_Studio_v4_Free
chmod +x start_linux_mac.sh
./start_linux_mac.sh
```

## 2. Install the free Kokoro voice one time

In the application's **Natural voice director**, click **DOWNLOAD FREE VOICE MODEL**. Or in the program's virtual environment run:

```bash
python setup_free_voice.py
```

The script downloads `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin` from the upstream `model-files-v1.1` release on GitHub, verifies their SHA-256 hashes, and stores them in `models/`. If GitHub downloads are blocked, obtain those **exact** files manually from [the official release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1), place them in `models/`, and retry. It will verify them before use.

Click **HEAR FREE NEURAL VOICE** to listen to the real result *on your machine* before rendering a full video. **There is no robotic fallback.** If the model is missing or can't run, you see an error instead of a bad-quality voice.

## 3. Make videos

1. Choose **Would You Rather**, **This or That**, or **Pick One**.
2. Use the free offline question bank or write your own creative questions.
3. Pick an original visual theme, 9:16 or 16:9, and optional custom images.
4. Choose an American or British voice. Adjust rate; the **Pacing profile** changes speed subtly (not expressive emotion).
5. Use **Preview a round**, then **RENDER VIDEO**. Narration duration is measured; scenes grow as necessary to avoid cutting off speech.
6. For music, import a track you've been granted the right to use. The built-in music library can match documented mood tags automatically. You can find suitable free tracks in [YouTube Studio Audio Library](https://www.youtube.com/audiolibrary) and download them through the official interface. Some tracks require attribution.
7. Inspect the resulting MP4 and narration; only publish content you've checked for originality, clarity, music rights, and platform compliance.

## 4. Output files

- `video_final.mp4`: rendered H.264/AAC video including audio
- `video_clean.mp4`: picture-only video for editing
- `narration.wav`, `voice_clips/`: separately editable narration
- `narration.srt` and `narration_cues.json`: captions and measured speech timestamps
- `original_sfx.wav`, `timeline.csv`, `thumbnail.png`, `description.txt`
- `project.json`, `project_portable.json`, `MUSIC_LICENSE.txt`
- A complete export ZIP

The model files are **not** copied into video exports.

## 5. CLI mode

```bash
python renderer.py --project examples/natural_voice_project.json --output outputs/my_first_video --quality Preview
```

For Full HD: `--quality 'Full HD'`. Adjust the project JSON questions first.

## 6. Test it

```bash
python -m pytest -q
```

Our automated tests verify video/audio encoding, local TTS integration through a mocked model, checksum-verified model downloading, timing, captions, motion, licensed music handling, and application UI loading. The test suite **does not** verify subjective humanlike audio quality, real model execution, or actual Windows/macOS results. GitHub Actions workflow contains test jobs for all 3 OSes but was not run here.

## License and economics

- ChoiceForge Python source: see `LICENSE`.
- Kokoro weights: Apache-2.0; kokoro-onnx runtime: MIT; downstream dependencies have their own licenses.
- FFmpeg and Python: no license fee, but obey relevant license terms if redistributing binaries.
- No API request from ChoiceForge's TTS; no paid services used.
- You may still have ordinary **internet, electricity, computer, or storage costs**. The first-time download needs internet.
- A free narrator does not guarantee YouTube monetization. Avoid mass-produced near-identical videos and follow YouTube partner program country rules.

## Troubleshooting

- `Free Kokoro model is not installed`: click the model-download button or run `python setup_free_voice.py`.
- `Checksum mismatch`: re-download exclusively from the official upstream release. Do not bypass verification.
- `No module named kokoro_onnx`: activate the local venv and run `pip install -r requirements.txt`.
- `ffmpeg not found`: install FFmpeg, close/reopen terminal, rerun.
- Startup fails on Python 3.14: install Python 3.12 and recreate `.venv`.
- Poor pronunciation: choose another Kokoro voice or rewrite the line with punctuation and simpler words. Free neural voices can still sound synthetic at times.
- Render slow on laptop: try `Preview`, reduce questions, then export Full HD.
