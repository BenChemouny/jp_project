from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import re
from typing import Any


NOUN_COLOR = "#FFFFFF"
VERB_ADJECTIVE_COLOR = "#A8E6CF"
FUNCTION_COLOR = "#DCEDC1"
FURIGANA_COLOR = "#888888"

_KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_KANA_RE = re.compile(r"^[\u3040-\u309f\u30a0-\u30ff\u30fc]+$")
_PUNCT_RE = re.compile(
    r"^[\s\u3000、。，．・？！!?\-ー「」『』（）()［］\[\]【】….,:;]+$"
)


@dataclass(frozen=True)
class JapaneseToken:
    surface: str
    part_of_speech: str
    reading: str
    furigana: str
    color: str
    furigana_color: str = FURIGANA_COLOR

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def analyze_japanese_text(text: str) -> list[JapaneseToken]:
    tokenizer, split_mode = _sudachi()
    tokens: list[JapaneseToken] = []
    for morpheme in tokenizer.tokenize(text, split_mode):
        surface = morpheme.surface()
        pos_parts = tuple(str(part) for part in morpheme.part_of_speech())
        reading = str(morpheme.reading_form() or "")
        part_of_speech = _normalize_part_of_speech(pos_parts)
        furigana = _furigana(surface, reading)
        tokens.append(
            JapaneseToken(
                surface=surface,
                part_of_speech=part_of_speech,
                reading=reading,
                furigana=furigana,
                color=_color_for_part_of_speech(part_of_speech),
            )
        )
    return tokens


def analyze_japanese_text_dicts(text: str) -> list[dict[str, str]]:
    return [token.to_dict() for token in analyze_japanese_text(text)]


@lru_cache(maxsize=1)
def _sudachi() -> tuple[Any, Any]:
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError:
        try:
            from sudachipy import dictionary, tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Japanese token rendering requires sudachipy and sudachidict_full. "
                "Run `uv sync --project client` before starting the UI."
            ) from exc

        dictionary_factory = dictionary.Dictionary
        split_mode = tokenizer.Tokenizer.SplitMode.C
    else:
        dictionary_factory = Dictionary
        split_mode = SplitMode.C

    for kwargs in ({"dict": "full"}, {"dict_type": "full"}, {}):
        try:
            return dictionary_factory(**kwargs).create(), split_mode
        except TypeError:
            continue

    return dictionary_factory().create(), split_mode


def _normalize_part_of_speech(pos_parts: tuple[str, ...]) -> str:
    major = pos_parts[0] if pos_parts else ""
    minor = pos_parts[1] if len(pos_parts) > 1 else ""

    if major == "名詞" and minor == "数詞":
        return "base_numeral"
    if major in {"名詞", "代名詞"}:
        return "noun"
    if major == "動詞":
        return "verb"
    if major == "形容詞":
        return "adjective"
    if major == "助詞":
        return "particle"
    if major == "助動詞":
        return "auxiliary_suffix"
    if major == "接続詞":
        return "conjunction"
    if major in {"補助記号", "空白"}:
        return "punctuation"
    return "other"


def _color_for_part_of_speech(part_of_speech: str) -> str:
    if part_of_speech in {"verb", "adjective"}:
        return VERB_ADJECTIVE_COLOR
    if part_of_speech in {"particle", "auxiliary_suffix", "conjunction"}:
        return FUNCTION_COLOR
    return NOUN_COLOR


def _furigana(surface: str, reading: str) -> str:
    if not surface or not reading or reading == "*":
        return ""
    if not _contains_kanji(surface):
        return ""
    if _is_pure_kana(surface) or _is_punctuation(surface):
        return ""
    return _to_hiragana(reading)


def _contains_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text))


def _is_pure_kana(text: str) -> bool:
    return bool(_KANA_RE.fullmatch(text))


def _is_punctuation(text: str) -> bool:
    return bool(_PUNCT_RE.fullmatch(text))


def _to_hiragana(text: str) -> str:
    try:
        import wanakana
    except ImportError as exc:
        raise RuntimeError(
            "Japanese furigana rendering requires wanakana. "
            "Run `uv sync --project client` before starting the UI."
        ) from exc

    converter = getattr(wanakana, "to_hiragana", None) or getattr(
        wanakana, "toHiragana", None
    )
    if callable(converter):
        return str(converter(text))

    # Keep rendering functional if the package exposes a different API version.
    return "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ン" else char for char in text)
