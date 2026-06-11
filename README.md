# Qwen3-ASR WebSocket Server

This repository contains a server-side voice receiver for speech-only PCM audio streams. It accepts WebSocket clients on `/ws/audio`, buffers each active speech segment, repeatedly sends the full current buffer to Qwen3-ASR for partial transcripts, and runs one final ASR pass on `speech_end`.

## Setup

Install the runtime dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Set the model path if you are using a local checkpoint:

```bash
export QWEN3_ASR_MODEL_PATH=/path/to/qwen3-asr-model
```

## Run

```bash
python -m server.audio_ws_server
```

Defaults:

- `HOST=0.0.0.0`
- `PORT=8000`
- `ASR_INTERVAL_MS=500`
- `SAMPLE_RATE=16000`
- `CHANNELS=1`
- `AUDIO_FORMAT=pcm_s16le`
- `DEVICE=cuda`
- `MAX_SEGMENT_SECONDS=30`
- `SEND_PARTIALS_TO_CLIENT=true`

The server listens at:

```text
ws://localhost:8000/ws/audio
```

## Client Protocol

Start a speech segment with a text JSON message:

```json
{
  "type": "speech_start",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le",
  "frame_ms": 30
}
```

Then send binary WebSocket messages containing raw `pcm_s16le` audio frames.

End the segment with:

```json
{
  "type": "speech_end"
}
```

Partial responses look like:

```json
{
  "type": "partial_transcript",
  "text": "...",
  "buffer_ms": 1230
}
```

Final responses look like:

```json
{
  "type": "final_transcript",
  "text": "...",
  "buffer_ms": 2450
}
```
