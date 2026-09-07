# SPDX-License-Identifier: Apache-2.0
# Adapted from rednote-hilab/dots.tts (Apache-2.0), dots_tts/utils/text.py.
"""Shared dots language tags without eagerly importing optional Pynini.

WeTextProcessing needs OpenFst, which has no macOS arm64 PyPI wheel. Normal
inference does not normalize text; import that optional dependency only when
normalization is requested, preserving the upstream normalizer itself.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingua import Language


@lru_cache(maxsize=1)
def get_language_detector():
    from lingua import Language, LanguageDetectorBuilder

    supported_languages = tuple(
        sorted(Language.all(), key=lambda language: language.name)
    )
    return LanguageDetectorBuilder.from_languages(*supported_languages).build()


def _lingua_language_to_code(language: Language | None) -> str | None:
    if language is None:
        return None
    iso_code_639_1 = getattr(language.iso_code_639_1, "name", None)
    if iso_code_639_1:
        return iso_code_639_1.lower()
    iso_code_639_3 = getattr(language.iso_code_639_3, "name", None)
    if iso_code_639_3:
        return iso_code_639_3.lower()
    return language.name.lower()


def detect(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    language = get_language_detector().detect_language_of(stripped)
    return _lingua_language_to_code(language)


def normalize_language_code(language: str | None) -> str | None:
    from langcodes import Language as LangcodesLanguage

    if language is None:
        return None

    stripped = language.strip()
    if not stripped or stripped.lower() in {"none", "unknown"}:
        return None
    if stripped.startswith("口音:"):
        return stripped

    for resolver in (LangcodesLanguage.get, LangcodesLanguage.find):
        try:
            normalized_language = resolver(stripped).prefer_macrolanguage()
        except Exception:
            continue

        language_code = (normalized_language.language or "").strip().upper()
        if language_code and language_code != "UND":
            return language_code
    return None


def attach_language_tag(text: str, language: str | None) -> str:
    if not text:
        return text

    language_code = normalize_language_code(language)
    if language_code is None:
        return text

    if language_code == "YUE":
        language_code = "口音:粤语"

    language_tag = f"[{language_code}]"
    if text.startswith(language_tag):
        return text
    return f"{language_tag}{text}"


def normalize_text(text: str) -> str:
    try:
        from dots_tts.utils.text import normalize_text as upstream_normalize
    except ImportError as exc:
        raise ValueError(
            "dots.tts normalize_text=True requires WeTextProcessing and Pynini; "
            "install them or provide normalized text with normalize_text=False"
        ) from exc
    return upstream_normalize(text)
