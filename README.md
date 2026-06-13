# Voice Streaming ASR Project

This project currently contains a two-part speech streaming prototype:

- `server/`: a WebSocket audio receiver that buffers speech segments and runs Qwen3-ASR for realtime partial and final transcripts.
- `client/`: a Raspberry Pi 5 compatible microphone streamer that performs local VAD and sends only active speech frames to the server.

The active audio protocol is raw mono `pcm_s16le` at 16 kHz over WebSocket binary messages, with JSON text messages for `speech_start` and `speech_end`.

## Current Status

Implemented:

- Server WebSocket endpoint at `ws://HOST:PORT/ws/audio`.
- Per-connection speech segment buffers with periodic full-buffer partial ASR.
- Final ASR pass on `speech_end`.
- Self-contained Qwen3-ASR adapter inside the server package.
- Raspberry Pi oriented client capture loop using `sounddevice`.
- Client-side high-pass filtering, mild noise reduction, pre-roll, hangover, and VAD state machine.
- Client-side one-second audio/VAD metric summaries for tuning speech detection and noise reduction.
- Optional Pygame local display UI for transcript, status, and metrics.
- Client VAD backends: Silero ONNX when configured, WebRTC VAD when installed, energy fallback otherwise.
- Server-side received-audio metrics and ASR latency/real-time-factor logging.
- Root `uv` workspace with separate `client` and `server` uv projects.

Not yet verified on hardware in this environment:

- Live microphone capture on Raspberry Pi 5.
- Qwen3-ASR model loading and GPU inference.
- End-to-end WebSocket streaming with real audio.

## Server Setup

On the desktop ASR machine:

```bash
uv sync --project server
export QWEN3_ASR_MODEL=/path/to/qwen3-asr-model
export QWEN3_ASR_DEVICE=cuda:0
```

Run the server:

```bash
uv run --project server jp-asr-server
```

Useful server settings:

- `HOST=0.0.0.0`
- `PORT=8000`
- `ASR_INTERVAL_MS=500`
- ASR defaults are `Qwen/Qwen3-ASR-1.7B`, `cuda:0`, `bfloat16`, batch `32`, max new tokens `256`, language `Japanese`, and `sdpa` attention.
- `MAX_SEGMENT_SECONDS=30`
- `SEND_PARTIALS_TO_CLIENT=true`

More detail is in `server/README.md`.

## Client Setup

On the Raspberry Pi 5:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
uv sync --project client
```

For Silero ONNX VAD:

```bash
uv sync --project client --extra silero
export SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx
```

Run the client:

```bash
export SERVER_WS_URL=ws://SERVER_HOST:8000/ws/audio
uv run --project client jp-voice-client
```

Run the client with the local display UI:

```bash
uv run --project client jp-voice-client-ui
```

Useful client settings:

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

More detail is in `client/README.md`.

## Run Order

1. Start the server on the desktop machine.
2. Confirm the server is listening on `ws://SERVER_HOST:8000/ws/audio`.
3. Start the client on the Raspberry Pi with `SERVER_WS_URL` pointing at the desktop.
4. Speak into the Pi microphone. The client logs VAD state and sends speech frames only while a segment is active. The server logs partial and final transcripts.
