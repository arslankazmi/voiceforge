"""``voiceforge`` — portable ``.voice`` artifacts: forge a voice from a clip, then
play it, or serve it (per-voice server / production backend).

    voiceforge forge  <clip> [-o out.voice] [--engine turbo|base] [--name N] [--serve]
    voiceforge say    <artifact.voice> "text" [-o out.wav] [--play] [--temperature T]
    voiceforge serve  <artifact.voice> [--host H] [--port P] [--token TOK]   # Tier A
    voiceforge backend [--host H] [--port P] [--token TOK]                    # Tier B
    voiceforge export-bundle <artifact.voice> [-o dir]

Heavy work (model load, synthesis) is lazy-imported so this module imports without
torch/chatterbox present. Everything routes through the single-residency
``heavy_session`` so it can never OOM the host.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC
from pathlib import Path
from typing import NoReturn

_ENGINE_ALIASES = {"turbo": "chatterbox-turbo", "base": "chatterbox"}


def _fail(msg: str) -> NoReturn:
    print(f"voiceforge: error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _audio_duration(path: str) -> float | None:
    """Best-effort clip duration in seconds (soundfile → wave → None)."""
    try:
        import soundfile as sf  # noqa: PLC0415

        return float(sf.info(path).duration)
    except Exception:
        pass
    try:
        import contextlib  # noqa: PLC0415
        import wave  # noqa: PLC0415

        with contextlib.closing(wave.open(path, "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def _chatterbox_version() -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("chatterbox-tts")
    except Exception:
        return ""


def _now_iso() -> str:
    from datetime import datetime  # noqa: PLC0415

    return datetime.now(UTC).isoformat()


def _play(path: str) -> None:
    player = "afplay" if sys.platform == "darwin" else "aplay"
    try:
        subprocess.run([player, path], check=False)  # noqa: S603
    except FileNotFoundError:
        print(f"(no {player} on PATH — saved to {path})")


def _cmd_forge(args: argparse.Namespace) -> None:
    from ._model_manager import heavy_session  # noqa: PLC0415
    from .engine import loader_for  # noqa: PLC0415
    from .validation import validate_audio  # noqa: PLC0415
    from .voice_artifact import VoiceMeta, forge  # noqa: PLC0415
    from .voice_library import slugify  # noqa: PLC0415

    engine = _ENGINE_ALIASES[args.engine]
    clip = args.clip
    if not Path(clip).exists():
        _fail(f"clip not found: {clip}")
    try:
        validate_audio(clip)
    except ValueError as exc:
        _fail(str(exc))

    if engine == "chatterbox-turbo":
        dur = _audio_duration(clip)
        if dur is not None and dur <= 5.0:
            _fail(
                f"Turbo requires a reference clip > 5s (got {dur:.1f}s). "
                "Use a ~6s+ clip, or --engine base."
            )

    name = args.name or Path(clip).stem
    meta = VoiceMeta(
        name=name,
        engine=engine,
        exaggeration=args.exaggeration,
        chatterbox_version=_chatterbox_version(),
        created=_now_iso(),
    )

    with heavy_session(engine, loader_for(engine)) as model:
        if model is None:
            _fail(
                "could not load the model (not installed or low memory). "
                "Install with: pip install 'voiceforge[clone]'"
            )
        meta.sample_rate = int(getattr(model, "sr", 24000))
        blob = forge(model, clip, meta)

    out = args.out or f"{slugify(name)}.voice"
    Path(out).write_bytes(blob)
    print(f"Forged {out}  ({len(blob) / 1024:.0f} KB)  engine={engine}  name={name!r}")

    if args.serve:
        _run_player_server(out, host="127.0.0.1", port=8080, token=None)


def _cmd_say(args: argparse.Namespace) -> None:
    from ._chatter_common import supported_generate_kwargs, tensor_to_wav  # noqa: PLC0415
    from ._model_manager import heavy_session  # noqa: PLC0415
    from .engine import loader_for  # noqa: PLC0415
    from .voice_artifact import apply, ensure_compatible, load  # noqa: PLC0415

    if not Path(args.artifact).exists():
        _fail(f"artifact not found: {args.artifact}")
    cond, meta = load(args.artifact)
    engine = meta.engine
    temperature = args.temperature if args.temperature is not None else meta.temperature

    with heavy_session(engine, loader_for(engine)) as model:
        if model is None:
            _fail("could not load the model (not installed or low memory).")
        ensure_compatible(meta, engine)
        apply(model, cond)
        kwargs = supported_generate_kwargs(model.generate, temperature=temperature)
        audio = model.generate(args.text, **kwargs)
        wav = tensor_to_wav(audio, model.sr)

    if wav is None:
        _fail("synthesis produced no audio.")
    Path(args.out).write_bytes(wav)
    print(f"Wrote {args.out}  ({len(wav) / 1024:.0f} KB)  voice={meta.name!r}")
    if args.play:
        _play(args.out)


def _run_player_server(artifact: str, host: str, port: int, token: str | None) -> None:
    import uvicorn  # noqa: PLC0415

    os.environ["VOICEFORGE_ARTIFACT"] = str(artifact)
    if token:
        os.environ["VOICEFORGE_TOKEN"] = token
    print(f"Serving voice {artifact} at http://{host}:{port}  (admin at /)")
    uvicorn.run("voiceforge.player_server:app", host=host, port=port, reload=False)


def _cmd_serve(args: argparse.Namespace) -> None:
    if not Path(args.artifact).exists():
        _fail(f"artifact not found: {args.artifact}")
    _run_player_server(args.artifact, args.host, args.port, args.token)


def _cmd_backend(args: argparse.Namespace) -> None:
    import uvicorn  # noqa: PLC0415

    if args.token:
        os.environ["AUTH_TOKEN"] = args.token
    print(
        f"Serving voice backend at http://{args.host}:{args.port}  "
        "(recording studio at /, docs at /docs)"
    )
    uvicorn.run("voiceforge.voice_backend:app", host=args.host, port=args.port, reload=False)


def _cmd_export_bundle(args: argparse.Namespace) -> None:
    import shutil  # noqa: PLC0415

    src = Path(args.artifact)
    if not src.exists():
        _fail(f"artifact not found: {src}")
    out_dir = Path(args.out or f"{src.stem}-bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out_dir / "voice.voice")
    (out_dir / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Self-contained voice server. Requires: pip install 'voiceforge[clone]'\n"
        'cd "$(dirname "$0")"\n'
        'exec voiceforge serve voice.voice "$@"\n'
    )
    (out_dir / "run.sh").chmod(0o755)
    (out_dir / "README.txt").write_text(
        "This folder is a self-contained voice.\n\n"
        "  ./run.sh            # start its browser voice server at http://127.0.0.1:8080\n"
        '  voiceforge say voice.voice "hello"   # or just render a clip\n\n'
        "Needs: pip install 'voiceforge[clone]' (pulls torch + chatterbox-tts).\n"
    )
    print(f"Wrote runnable voice bundle → {out_dir}/  (run.sh, voice.voice, README.txt)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voiceforge",
        description="Portable .voice artifacts — forge a voice from a clip, then play or serve it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("forge", help="Clip → portable .voice artifact.")
    p.add_argument("clip", help="Reference audio clip (Turbo needs > 5s).")
    p.add_argument("-o", "--out", help="Output .voice path (default: <name>.voice).")
    p.add_argument("--engine", choices=["turbo", "base"], default="turbo")
    p.add_argument("--name", help="Voice name (default: clip filename).")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--serve", action="store_true", help="Spin up its server after forging.")
    p.set_defaults(func=_cmd_forge)

    p = sub.add_parser("say", help="Artifact + text → WAV (no clip needed).")
    p.add_argument("artifact", help="A .voice file.")
    p.add_argument("text", help="Text to speak.")
    p.add_argument("-o", "--out", default="out.wav")
    p.add_argument("--play", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.set_defaults(func=_cmd_say)

    p = sub.add_parser("serve", help="Tier A: per-voice server + browser admin page.")
    p.add_argument("artifact", help="A .voice file.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--token", default=None, help="Optional bearer token to require.")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("backend", help="Tier B: production multi-voice backend service.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default=None, help="Optional bearer token (also AUTH_TOKEN env).")
    p.set_defaults(func=_cmd_backend)

    p = sub.add_parser(
        "export-bundle", help="Emit a runnable folder carrying the voice + launcher."
    )
    p.add_argument("artifact", help="A .voice file.")
    p.add_argument("-o", "--out", help="Output directory (default: <name>-bundle).")
    p.set_defaults(func=_cmd_export_bundle)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
