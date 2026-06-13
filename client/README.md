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

For the preferred Silero ONNX VAD path, install `onnxruntime` and `numpy`.
The client uses `client/models/silero_vad.onnx` by default when that file exists:

```bash
uv sync --project client --extra silero
```

Set `SILERO_VAD_ONNX_PATH` or pass `--silero-vad-onnx-path` to use a different model file.

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
- `DYNAMIC_VAD=true`
- `VAD_MIN_START_THRESHOLD=0.35`
- `VAD_MIN_CONTINUE_THRESHOLD=0.25`
- `VAD_NOISE_MARGIN_DB=6.0`
- `VAD_SPEECH_MARGIN_DB=12.0`
- `VAD_ENERGY_FALLBACK=true`
- `VAD_ENERGY_START_MARGIN_DB=9.0`
- `VAD_ENERGY_CONTINUE_MARGIN_DB=13.0`
- `ENABLE_NOISE_REDUCTION=true`
- `HIGH_PASS_HZ=100`
- `INPUT_DEVICE=0`
- `VAD_BACKEND=auto`

Dynamic VAD lowers the effective Silero/WebRTC/energy thresholds when speech is clearly above the local noise floor and raises them when ambient noise or model background probability rises. Energy fallback lets foreground audio open the speech gate even if the model probability stays near zero. Use `--no-dynamic-vad` or `--no-vad-energy-fallback` to disable those behaviors.

List audio devices with:

```bash
uv run --project client python -m sounddevice
```

## Audio Metrics

The client logs a one-second audio/VAD summary while running:

```text
[audio] frames=34 raw=-42.1dBFS filtered=-43.0dBFS out=-45.8dBFS out_min=-52.0dBFS out_max=-30.4dBFS peak=-16.2dBFS clip=0 floor=-58.4dBFS snr=15.4dB nr=-2.8dB vad=0.72 thr=0.35/0.25 vad_pos=64% state=streaming streaming=True segment_ms=1230 ws=connected
```

- `raw`, `filtered`, and `out` are average RMS levels before filtering, after high-pass filtering, and after noise reduction.
- `peak` and `clip` help diagnose microphone gain problems.
- `floor` is the current adaptive noise floor estimate, and `nr` is the average noise-reduction gain applied during the window.
- `snr` is the VAD input level above the adaptive floor.
- `vad` is the average VAD probability, `thr` is the current start/continue threshold pair, and `vad_pos` is the share of frames above the continue threshold.

## Display UI

The Pygame UI renders the latest transcript only. Incoming transcript text is tokenized with SudachiPy using the full Sudachi dictionary. Each token carries its raw surface text, normalized part-of-speech tag, native Katakana reading, and optional Hiragana furigana converted with wanakana.

Tokens that contain Kanji render furigana above the base text. Pure Kana tokens and punctuation keep an empty furigana field so the UI does not duplicate readings. Base text uses this palette:

- Nouns and base numerals: `#FFFFFF`
- Verbs and adjectives: `#A8E6CF`
- Particles, auxiliary suffixes, and conjunctions: `#DCEDC1`
- Furigana: `#888888`

Status and compact audio/VAD metrics appear in the top-left corner.

Keyboard controls:

- `W` / `A` / `S` / `D`: move transcript position.
- `Q` / `E`: rotate transcript.
- `Z` / `C`: skew transcript.
- `F` / `V`: increase or decrease transcript scale.
- `R`: reset transcript position, rotation, scale, and skew.
- `B`: toggle bold transcript text.
- `P`: pause or resume recording. Pausing closes any active stream and drops microphone frames locally.
- `I`: toggle detailed status and metrics.
- `H`: toggle the keyboard help window.
