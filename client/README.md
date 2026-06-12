# Raspberry Pi Voice Streaming Client

This client captures microphone audio on Linux, converts it to mono 16 kHz `pcm_s16le`, applies a light high-pass filter and mild noise reduction, runs local VAD, and streams only active speech to the server WebSocket endpoint.

## Install

On Raspberry Pi OS, install PortAudio first:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
```

Install Python dependencies with uv:

```bash
uv sync --project client
```

For the preferred Silero ONNX VAD path, install `onnxruntime` and `numpy`, then provide a local ONNX model:

```bash
uv sync --project client --extra silero
export SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx
```

If Silero is not configured, the client tries WebRTC VAD. If WebRTC VAD is unavailable, it falls back to a simple energy VAD.

## Run

```bash
export SERVER_WS_URL=ws://SERVER_HOST:8000/ws/audio
uv run --project client jp-voice-client
```

Run with the local display UI:

```bash
export SERVER_WS_URL=ws://SERVER_HOST:8000/ws/audio
uv run --project client jp-voice-client-ui
```

Useful settings:

- `SAMPLE_RATE=16000`
- `FRAME_MS=30`
- `PRE_ROLL_MS=500`
- `HANGOVER_MS=1000`
- `MIN_SPEECH_MS=300`
- `VAD_START_THRESHOLD=0.65`
- `VAD_CONTINUE_THRESHOLD=0.45`
- `ENABLE_NOISE_REDUCTION=true`
- `HIGH_PASS_HZ=100`
- `INPUT_DEVICE=0`
- `VAD_BACKEND=auto`

List audio devices with:

```bash
uv run --project client python -m sounddevice
```

## Audio Metrics

The client logs a one-second audio/VAD summary while running:

```text
[audio] frames=34 raw=-42.1dBFS filtered=-43.0dBFS out=-45.8dBFS out_min=-52.0dBFS out_max=-30.4dBFS peak=-16.2dBFS clip=0 floor=-58.4dBFS nr=-2.8dB vad=0.72 vad_pos=64% state=streaming streaming=True segment_ms=1230 ws=connected
```

- `raw`, `filtered`, and `out` are average RMS levels before filtering, after high-pass filtering, and after noise reduction.
- `peak` and `clip` help diagnose microphone gain problems.
- `floor` is the current adaptive noise floor estimate, and `nr` is the average noise-reduction gain applied during the window.
- `vad` is the average VAD probability, and `vad_pos` is the share of frames above the continue threshold.

## Display UI

The Pygame UI renders the latest transcript only. Partial transcripts are grey, and final transcripts are white. Status and compact audio/VAD metrics appear in the top-left corner.

Keyboard controls:

- `W` / `A` / `S` / `D`: move transcript position.
- `Q` / `E`: rotate transcript.
- `Z` / `C`: decrease or increase transcript scale.
