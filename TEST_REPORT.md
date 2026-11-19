# ChoiceForge Studio v4 — Test report

Tested on Linux, Python 3.13, FFmpeg available, in an environment without outbound model download access.

- **35 local automated unit/integration tests passed** including video and audio encoding, speech-timing, separate stems, portrait and landscape, animation, licensable-music selection, model download integrity, model-provider interface via a fake Kokoro runtime, and rejection of paid/robotic providers.
- Video encoding tests use FFmpeg/ffprobe to confirm H.264 video and AAC audio in real exported MP4s. Captions and scene durations are checked.
- UI imports and mounts successfully in the test environment. End-to-end **genuine Kokoro voice was NOT tested**, because the model files could not be downloaded into the environment.
- Only the Linux test environment was used. Windows and macOS startup scripts and a GitHub Actions CI matrix are included but were not executed locally.
- No subscriptions, paid APIs, account credentials or hidden robotic TTS are required.

## Remaining live checks on your laptop

1. Download both Kokoro files through the installer and verify hashes.
2. Audition US Bella, US Heart and UK Emma at different speaking rates.
3. Render a one-question video and listen to it fully, then check the preview and MP4 sound quality.
4. Check that the app opens and FFmpeg is on PATH on your particular OS.
5. Listen for mispronunciations (including proper names), pacing, and sudden volume changes; fix text and rerender as needed.

These checks cannot be certified without running the actual neural model on your computer. Natural sounding is a qualitative property and no model is guaranteed to sound perfectly human in all contexts.
