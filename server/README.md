# Qwen3-ASR WebSocket Server

The server receives speech-only PCM audio streams over WebSocket at `/ws/audio`, buffers each active speech segment, repeatedly sends the full current buffer to Qwen3-ASR for partial transcripts, and runs one final ASR pass on `speech_end`.

## Setup

Install the server runtime dependencies with uv:

```bash
uv sync --project server
```

Set the model path if you are using a local checkpoint:

```bash
export QWEN3_ASR_MODEL=/path/to/qwen3-asr-model
export QWEN3_ASR_DEVICE=cuda:0
```

## Run

```bash
uv run --project server jp-asr-server
```

Defaults:

- `HOST=0.0.0.0`
- `PORT=8000`
- `ASR_INTERVAL_MS=500`
- `SAMPLE_RATE=16000`
- `CHANNELS=1`
- `AUDIO_FORMAT=pcm_s16le`
- ASR defaults are `Qwen/Qwen3-ASR-1.7B`, `cuda:0`, `bfloat16`, batch `32`, max new tokens `256`, language `Japanese`, and `sdpa` attention.
- `MAX_SEGMENT_SECONDS=30`
- `SEND_PARTIALS_TO_CLIENT=true`
- `MIN_ASR_SEGMENT_MS=650`
- `MIN_ASR_RMS_DBFS=-55.0`
- `MAX_ASR_SILENCE_RATIO=0.98`
- `MAX_ASR_LOW_LEVEL_RATIO=0.995`
- `SUPPRESS_SHORT_FILLERS=true`

The server listens at:

```text
ws://localhost:8000/ws/audio
```

## WebSocket Protocol

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
  "buffer_ms": 1230,
  "asr_latency_ms": 210,
  "asr_rtf": 0.17
}
```

Final responses look like:

```json
{
  "type": "final_transcript",
  "text": "...",
  "buffer_ms": 2450,
  "asr_latency_ms": 412,
  "asr_rtf": 0.17
}
```

The server may reject likely non-speech before ASR. In that case it sends an empty final transcript with a rejection reason such as `rejected_too_short`, `rejected_low_rms`, `rejected_mostly_silence`, or `rejected_low_level`.

## Audio Metrics

The server logs received-audio and ASR timing metrics for partial and final passes:

```text
[final 2450ms] frames=82 rms=-31.8dBFS frame_rms=-35.0dBFS rms_min=-60.0dBFS rms_max=-24.2dBFS peak=-5.2dBFS clip=0.00% silence=18% low=31% asr=412ms rtf=0.17 ...
```

- `rms` is the segment RMS, while `frame_rms`, `rms_min`, and `rms_max` summarize received frame levels.
- `peak`, `clip`, `silence`, and `low` help identify clipping, dead air, and low-level audio arriving from the client.
- `asr` is inference latency for that pass, and `rtf` is real-time factor.

## Noise and Click Rejection

Before final ASR, the server rejects segments that are too short, too quiet, mostly silence, or mostly low-level samples according to the thresholds above. Partial ASR is also skipped until the current buffer passes the same checks.

If `SUPPRESS_SHORT_FILLERS=true`, short or weak segments that transcribe as common filler text such as `はい`, `うん`, or `ええ` are returned as empty final transcripts with reason `suppressed_short_filler`.
