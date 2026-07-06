"""Tests for the voiceforge CLI parser + guards (no real model loads)."""

from __future__ import annotations

import pytest

from voiceforge.cli import _ENGINE_ALIASES, build_parser, main


def test_forge_parser_defaults():
    args = build_parser().parse_args(["forge", "clip.wav"])
    assert args.command == "forge"
    assert args.engine == "turbo"
    assert args.exaggeration == 0.5


def test_engine_aliases():
    assert _ENGINE_ALIASES == {"turbo": "chatterbox-turbo", "base": "chatterbox"}


def test_say_parser():
    args = build_parser().parse_args(["say", "v.voice", "hello", "--temperature", "0.9"])
    assert args.command == "say"
    assert args.text == "hello"
    assert args.temperature == 0.9


def test_serve_and_backend_parsers():
    a = build_parser().parse_args(["serve", "v.voice", "--port", "9000", "--token", "x"])
    assert a.command == "serve" and a.port == 9000 and a.token == "x"
    b = build_parser().parse_args(["backend", "--host", "0.0.0.0"])
    assert b.command == "backend" and b.host == "0.0.0.0"


def test_forge_missing_clip_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["forge", str(tmp_path / "nope.wav")])


def test_say_missing_artifact_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["say", str(tmp_path / "nope.voice"), "hi"])
