from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from queue import Empty, Queue
import threading
from typing import Any

from client.config import load_config
from client.voice_stream_client import run_client


BACKGROUND = (0, 0, 0)
WHITE = (245, 245, 245)
GREY = (135, 135, 135)
DIM = (95, 95, 95)
GREEN = (67, 209, 114)
RED = (220, 76, 76)
YELLOW = (228, 188, 72)


@dataclass
class UiState:
    transcript: str = ""
    transcript_phase: str = "partial"
    connection_status: str = "disconnected"
    metrics: dict[str, float | int] = field(default_factory=dict)
    vad_state: str = "idle"
    streaming: bool = False
    segment_ms: int = 0
    text_x: float = 0.5
    text_y: float = 0.5
    rotation_deg: float = 0.0
    scale: float = 1.0


def main() -> None:
    config = load_config()
    run_ui_client(config)


def run_ui_client(config: Any) -> None:
    events: Queue[dict[str, Any]] = Queue()
    stop_requested = threading.Event()

    def publish(event: dict[str, Any]) -> None:
        events.put(event)

    client_thread = threading.Thread(
        target=_run_client_thread,
        args=(config, publish, stop_requested),
        daemon=True,
    )
    client_thread.start()

    try:
        _run_pygame_loop(events, stop_requested)
    finally:
        stop_requested.set()
        client_thread.join(timeout=3.0)


def _run_client_thread(
    config: Any,
    publish: Any,
    stop_requested: threading.Event,
) -> None:
    try:
        asyncio.run(
            run_client(
                config,
                event_sink=publish,
                install_signal_handlers=False,
                should_stop=stop_requested.is_set,
            )
        )
    except Exception as exc:
        publish({"type": "ui_error", "message": str(exc)})


def _run_pygame_loop(
    events: Queue[dict[str, Any]],
    stop_requested: threading.Event,
) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("JP Voice Client")
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    state = UiState()
    font_path = _find_font(pygame)

    while not stop_requested.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_requested.set()
            elif event.type == pygame.KEYDOWN:
                _handle_key(event.key, state, pygame)

        _drain_events(events, state)
        _draw(screen, state, font_path, pygame)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _handle_key(key: int, state: UiState, pygame: Any) -> None:
    move_step = 0.025
    if key == pygame.K_a:
        state.text_x -= move_step
    elif key == pygame.K_d:
        state.text_x += move_step
    elif key == pygame.K_w:
        state.text_y -= move_step
    elif key == pygame.K_s:
        state.text_y += move_step
    elif key == pygame.K_q:
        state.rotation_deg -= 3.0
    elif key == pygame.K_e:
        state.rotation_deg += 3.0
    elif key == pygame.K_z:
        state.scale = max(0.25, state.scale * 0.95)
    elif key == pygame.K_c:
        state.scale = min(4.0, state.scale * 1.05)
    state.text_x = min(1.2, max(-0.2, state.text_x))
    state.text_y = min(1.2, max(-0.2, state.text_y))


def _drain_events(events: Queue[dict[str, Any]], state: UiState) -> None:
    while True:
        try:
            event = events.get_nowait()
        except Empty:
            return
        event_type = event.get("type")
        if event_type == "connection":
            state.connection_status = str(event.get("status", "disconnected"))
        elif event_type == "transcript":
            state.transcript = str(event.get("text", ""))
            state.transcript_phase = str(event.get("phase", "partial"))
        elif event_type == "metrics":
            state.metrics = dict(event.get("metrics", {}))
            state.vad_state = str(event.get("state", "idle"))
            state.streaming = bool(event.get("streaming", False))
            state.segment_ms = int(event.get("segment_ms", 0))
            state.connection_status = str(event.get("ws", state.connection_status))
        elif event_type == "ui_error":
            state.connection_status = "error"
            state.transcript = str(event.get("message", ""))
            state.transcript_phase = "final"


def _draw(screen: Any, state: UiState, font_path: str | None, pygame: Any) -> None:
    width, height = screen.get_size()
    screen.fill(BACKGROUND)
    _draw_status_and_metrics(screen, state, font_path, pygame)
    _draw_transcript(screen, state, font_path, pygame, width, height)


def _draw_status_and_metrics(
    screen: Any,
    state: UiState,
    font_path: str | None,
    pygame: Any,
) -> None:
    font = pygame.font.Font(font_path, 17)
    status_color = GREEN if state.connection_status == "connected" else RED
    if state.connection_status == "error":
        status_color = YELLOW

    pygame.draw.circle(screen, status_color, (18, 18), 6)
    lines = [
        f"{state.connection_status} {state.vad_state} {'stream' if state.streaming else 'idle'} {state.segment_ms}ms",
        _metric_line(
            state.metrics,
            ("raw_rms_dbfs", "filtered_rms_dbfs", "output_rms_dbfs"),
            ("raw", "hp", "out"),
            "dB",
        ),
        _metric_line(
            state.metrics,
            ("peak_dbfs", "noise_floor_dbfs", "nr_gain_db"),
            ("pk", "floor", "nr"),
            "dB",
        ),
        _metric_line(
            state.metrics,
            ("vad_probability", "vad_positive_ratio", "clip_count"),
            ("vad", "pos", "clip"),
            "",
        ),
    ]

    y = 8
    for index, line in enumerate(lines):
        color = WHITE if index == 0 else DIM
        rendered = font.render(line, True, color)
        screen.blit(rendered, (32 if index == 0 else 12, y))
        y += 20


def _draw_transcript(
    screen: Any,
    state: UiState,
    font_path: str | None,
    pygame: Any,
    width: int,
    height: int,
) -> None:
    if not state.transcript:
        return

    font_size = max(24, int(74 * state.scale))
    font = pygame.font.Font(font_path, font_size)
    color = WHITE if state.transcript_phase == "final" else GREY
    max_width = int(width * 0.86)
    lines = _wrap_text(state.transcript, font, max_width)
    line_height = int(font.get_linesize() * 1.08)
    surface_width = max(font.size(line)[0] for line in lines)
    surface_height = line_height * len(lines)
    text_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

    for index, line in enumerate(lines):
        rendered = font.render(line, True, color)
        x = (surface_width - rendered.get_width()) // 2
        text_surface.blit(rendered, (x, index * line_height))

    rotated = pygame.transform.rotozoom(text_surface, -state.rotation_deg, 1.0)
    rect = rotated.get_rect(center=(int(width * state.text_x), int(height * state.text_y)))
    screen.blit(rotated, rect)


def _wrap_text(text: str, font: Any, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + char
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _metric_line(
    metrics: dict[str, float | int],
    keys: tuple[str, str, str],
    labels: tuple[str, str, str],
    suffix: str,
) -> str:
    parts: list[str] = []
    for key, label in zip(keys, labels, strict=True):
        value = metrics.get(key)
        if isinstance(value, float):
            if key == "vad_positive_ratio":
                parts.append(f"{label}:{value:.0%}")
            else:
                parts.append(f"{label}:{value:.1f}{suffix}")
        elif isinstance(value, int):
            parts.append(f"{label}:{value}")
        else:
            parts.append(f"{label}:--")
    return " ".join(parts)


def _find_font(pygame: Any) -> str | None:
    preferred = (
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Noto Sans CJK",
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "TakaoGothic",
        "DejaVu Sans",
    )
    for name in preferred:
        path = pygame.font.match_font(name)
        if path:
            return path
    return None


if __name__ == "__main__":
    main()
