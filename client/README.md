# Raspberry Pi Voice Streaming Client

This client captures microphone audio on Linux, converts it to mono 16 kHz `pcm_s16le`, applies a light high-pass filter and mild noise reduction, runs local VAD, and streams only active speech to the server WebSocket endpoint.

## Install

On Raspberry Pi OS, install PortAudio first:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev
```

Install Python dependencies:

```bash
python -m pip install -r client/requirements.txt
```

For the preferred Silero ONNX VAD path, install `onnxruntime` and `numpy`, then provide a local ONNX model:

```bash
python -m pip install onnxruntime numpy
export SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx
```

If Silero is not configured, the client tries WebRTC VAD. If WebRTC VAD is unavailable, it falls back to a simple energy VAD.

## Run

```bash
export SERVER_WS_URL=ws://SERVER_HOST:8000/ws/audio
python -m client.voice_stream_client
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
python -m sounddevice
```
