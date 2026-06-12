from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
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
PANEL = (18, 18, 18)
PANEL_BORDER = (70, 70, 70)
FONT_NAME = "Noto Sans CJK JP"


@dataclass(frozen=True)
class FontChoice:
    name: str
    path: str | None


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
    skew: float = 0.0
    font_bold: bool = False
    show_info: bool = False
    show_help: bool = False


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
    font_choice = _find_font(pygame)

    while not stop_requested.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_requested.set()
            elif event.type == pygame.KEYDOWN:
                _handle_key(event.key, state, pygame)

        _drain_events(events, state)
        _draw(screen, state, font_choice, pygame)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _handle_key(
    key: int,
    state: UiState,
    pygame: Any,
) -> None:
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
        state.skew = max(-0.8, state.skew - 0.04)
    elif key == pygame.K_c:
        state.skew = min(0.8, state.skew + 0.04)
    elif key == pygame.K_f:
        state.scale = min(4.0, state.scale * 1.05)
    elif key == pygame.K_v:
        state.scale = max(0.25, state.scale * 0.95)
    elif key == pygame.K_r:
        _reset_transform(state)
    elif key == pygame.K_b:
        state.font_bold = not state.font_bold
    elif key == pygame.K_i:
        state.show_info = not state.show_info
    elif key == pygame.K_h:
        state.show_help = not state.show_help
    state.text_x = min(1.2, max(-0.2, state.text_x))
    state.text_y = min(1.2, max(-0.2, state.text_y))


def _reset_transform(state: UiState) -> None:
    state.text_x = 0.5
    state.text_y = 0.5
    state.rotation_deg = 0.0
    state.scale = 1.0
    state.skew = 0.0


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


def _draw(
    screen: Any,
    state: UiState,
    font_choice: FontChoice,
    pygame: Any,
) -> None:
    width, height = screen.get_size()
    screen.fill(BACKGROUND)
    _draw_status_and_metrics(screen, state, font_choice, pygame)
    _draw_transcript(screen, state, font_choice, pygame, width, height)
    if state.show_help:
        _draw_help(screen, state, font_choice, pygame, width, height)


def _draw_status_and_metrics(
    screen: Any,
    state: UiState,
    font_choice: FontChoice,
    pygame: Any,
) -> None:
    font = _make_font(pygame, font_choice, state, 17, force_regular=True)
    status_color = GREEN if state.connection_status == "connected" else RED
    if state.connection_status == "error":
        status_color = YELLOW

    pygame.draw.circle(screen, status_color, (18, 18), 6)
    transcript_label = "final" if state.transcript_phase == "final" else "partial"
    lines = [
        f"{state.connection_status} {transcript_label} {state.vad_state} {'stream' if state.streaming else 'idle'}",
    ]
    if state.show_info:
        lines.extend(
            [
                f"segment:{state.segment_ms}ms font:{font_choice.name} bold:{state.font_bold}",
                f"pos:{state.text_x:.2f},{state.text_y:.2f} rot:{state.rotation_deg:.0f} scale:{state.scale:.2f} skew:{state.skew:.2f}",
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
        )
    else:
        lines.append("H:help I:info")

    y = 8
    for index, line in enumerate(lines):
        color = WHITE if index == 0 else DIM
        rendered = font.render(line, True, color)
        screen.blit(rendered, (32 if index == 0 else 12, y))
        y += 20


def _draw_help(
    screen: Any,
    state: UiState,
    font_choice: FontChoice,
    pygame: Any,
    width: int,
    height: int,
) -> None:
    font = _make_font(pygame, font_choice, state, 22, force_regular=True)
    small = _make_font(pygame, font_choice, state, 18, force_regular=True)
    lines = [
        "Keyboard",
        "W/A/S/D  move transcript",
        "Q/E      rotate transcript",
        "Z/C      skew transcript",
        "F/V      scale transcript",
        "R        reset transform",
        "B        toggle bold",
        "I        toggle detailed status/metrics",
        "H        toggle this help",
        "",
        f"font: {font_choice.name}",
        f"file: {_font_label(font_choice.path)}",
    ]
    rendered_lines = [
        (font if index == 0 else small).render(line, True, WHITE if index == 0 else GREY)
        for index, line in enumerate(lines)
    ]
    panel_width = max(item.get_width() for item in rendered_lines) + 48
    panel_height = sum(item.get_height() for item in rendered_lines) + 44
    panel = pygame.Rect(0, 0, panel_width, panel_height)
    panel.center = (width // 2, height // 2)
    pygame.draw.rect(screen, PANEL, panel, border_radius=6)
    pygame.draw.rect(screen, PANEL_BORDER, panel, width=1, border_radius=6)

    y = panel.y + 22
    for rendered in rendered_lines:
        screen.blit(rendered, (panel.x + 24, y))
        y += rendered.get_height()


def _font_label(path: str | None) -> str:
    if not path:
        return "pygame default"
    return Path(path).name


def _make_font(
    pygame: Any,
    font_choice: FontChoice,
    state: UiState,
    size: int,
    force_regular: bool = False,
) -> Any:
    font = pygame.font.Font(font_choice.path, size)
    if state.font_bold and not force_regular:
        font.set_bold(True)
    return font


def _draw_transcript(
    screen: Any,
    state: UiState,
    font_choice: FontChoice,
    pygame: Any,
    width: int,
    height: int,
) -> None:
    if not state.transcript:
        return

    font_size = max(24, int(74 * state.scale))
    font = _make_font(pygame, font_choice, state, font_size)
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

    skewed = _skew_surface(text_surface, state.skew, pygame)
    rotated = pygame.transform.rotozoom(skewed, -state.rotation_deg, 1.0)
    rect = rotated.get_rect(center=(int(width * state.text_x), int(height * state.text_y)))
    screen.blit(rotated, rect)


def _skew_surface(surface: Any, skew: float, pygame: Any) -> Any:
    if abs(skew) < 0.001:
        return surface

    width, height = surface.get_size()
    offset = int(abs(skew) * height)
    skewed = pygame.Surface((width + offset, height), pygame.SRCALPHA)
    for y in range(height):
        row = surface.subsurface((0, y, width, 1))
        shift = int(skew * (height - y))
        x = shift + offset if skew < 0 else shift
        skewed.blit(row, (x, y))
    return skewed


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


def _find_font(pygame: Any) -> FontChoice:
    return FontChoice(name=FONT_NAME, path=pygame.font.match_font(FONT_NAME))


if __name__ == "__main__":
    main()
