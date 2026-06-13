# Silero ONNX VAD Setup

The client can run Silero VAD locally on the Raspberry Pi 5 through ONNX Runtime. This gives better speech/noise separation than WebRTC VAD in many noisy rooms while keeping the heavier ASR work on the desktop server.

## Install

Install the normal client requirements first:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
uv sync --project client
```

Install the optional Silero dependencies:

```bash
uv sync --project client --extra silero
```

This installs `onnxruntime` and `numpy` for the client project.

## Get the Model

Download a Silero VAD ONNX model from the official `snakers4/silero-vad` project and place it somewhere stable on the Pi, for example:

```bash
mkdir -p models
# Save the downloaded ONNX file as:
# models/silero_vad.onnx
```

Keep the model outside generated build/cache folders so it survives dependency reinstalls.

## Run With Silero

Point the client at the ONNX file and select the Silero backend:

```bash
export SERVER_WS_URL=ws://SERVER_HOST:8000/ws/audio
export SILERO_VAD_ONNX_PATH=/home/pi/jp_project/models/silero_vad.onnx
export VAD_BACKEND=silero
uv run --project client jp-voice-client-ui
```

For a softer fallback path, use auto mode:

```bash
export VAD_BACKEND=auto
```

In auto mode, the client uses Silero only when `SILERO_VAD_ONNX_PATH` is set and the model loads successfully. Otherwise it falls back to WebRTC VAD, then to energy VAD.

## Recommended Pi 5 Settings

Start with:

```bash
export SAMPLE_RATE=16000
export FRAME_MS=30
export VAD_START_THRESHOLD=0.65
export VAD_CONTINUE_THRESHOLD=0.45
export MIN_SPEECH_MS=300
export HANGOVER_MS=1000
export MAX_CLIENT_SEGMENT_MS=10000
export DISCONNECTED_RESET_MS=1500
```

If random clicks or room noise still start speech, raise `VAD_START_THRESHOLD` in small steps:

```bash
export VAD_START_THRESHOLD=0.75
```

If speech cuts off too quickly, lower `VAD_CONTINUE_THRESHOLD` slightly or increase `HANGOVER_MS`:

```bash
export VAD_CONTINUE_THRESHOLD=0.35
export HANGOVER_MS=1300
```

## Verify It Is Active

When the client starts, look for:

```text
[vad] backend=silero
```

If you see `backend=webrtc` or `backend=energy`, Silero did not load. Check the model path and run once with `VAD_BACKEND=silero` so startup fails loudly instead of falling back.

## Troubleshooting

- `No such file`: confirm `SILERO_VAD_ONNX_PATH` is an absolute path on the Pi.
- `onnxruntime` import error: rerun `uv sync --project client --extra silero`.
- High CPU: keep `SAMPLE_RATE=16000`, keep the CPU ONNX provider, and avoid running other heavy UI or browser work on the Pi.
- VAD always positive: verify microphone gain first. In quiet room logs, `raw` and `out` should be far below normal speech. If idle noise is near `-20 dBFS`, reduce input gain or move the microphone before tuning thresholds.
- No speech detected: lower `VAD_START_THRESHOLD` and confirm the audio device is correct with `uv run --project client python -m sounddevice`.

