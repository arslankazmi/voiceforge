"""Portable ``.voice`` artifact — serialize and reload a Chatterbox speaker voice.

A Chatterbox clone is **zero-shot**: the "voice" is not trained weights, it is the
*speaker conditioning* the model derives from a reference clip. Chatterbox exposes that
as a ``Conditionals`` dataclass (``t3`` + ``gen`` tensors) with first-class ``save``/
``load`` — its own default voice ships as a ``conds.pt``. We wrap that into a single
portable ``.voice`` file so a voice can be forged once and reused/served forever,
**without the original clip**.

File format (one ``torch.save`` blob, loadable ``weights_only=True``)::

    {
      "meta": { "schema_version", "name", "engine", "sample_rate", ... },  # primitives
      "cond": { "t3": <T3Cond.__dict__ tensors>, "gen": <s3gen ref dict> }, # tensors
    }

Artifacts are **variant-locked**: a ``chatterbox`` voice cannot load into
``chatterbox-turbo`` (different tokenizer / prompt length / vocoder). ``load`` enforces it.

Torch + ``chatterbox-tts`` are only needed for :func:`forge` / :func:`load`. The metadata
layer (:class:`VoiceMeta`) is torch-free and independently unit-testable.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

#: Supported engine variants. An artifact is locked to the one it was forged with.
ENGINES = ("chatterbox", "chatterbox-turbo")

#: Filename extension for artifacts.
VOICE_EXT = ".voice"


class ArtifactError(Exception):
    """Raised for malformed artifacts or engine-variant mismatches."""


@dataclass
class VoiceMeta:
    """Metadata stamped into every ``.voice`` file (all primitives — torch-free)."""

    name: str
    engine: str
    sample_rate: int = 24000
    schema_version: int = SCHEMA_VERSION
    chatterbox_version: str = ""
    created: str = ""
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VoiceMeta:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


def validate_engine(engine: str) -> None:
    """Raise :class:`ArtifactError` if *engine* is not a known variant."""
    if engine not in ENGINES:
        raise ArtifactError(f"Unknown engine {engine!r}; expected one of {ENGINES}.")


def _cond_classes(engine: str):  # noqa: ANN201
    """Import the ``(Conditionals, T3Cond)`` classes for *engine* (lazy)."""
    validate_engine(engine)
    if engine == "chatterbox-turbo":
        from chatterbox.tts_turbo import Conditionals  # noqa: PLC0415
    else:
        from chatterbox.tts import Conditionals  # noqa: PLC0415
    from chatterbox.models.t3.modules.cond_enc import T3Cond  # noqa: PLC0415

    return Conditionals, T3Cond


def dumps(cond, meta: VoiceMeta) -> bytes:  # noqa: ANN001
    """Serialize a Chatterbox ``Conditionals`` + :class:`VoiceMeta` to ``.voice`` bytes."""
    import torch  # noqa: PLC0415

    validate_engine(meta.engine)
    blob = {"meta": meta.to_dict(), "cond": {"t3": dict(cond.t3.__dict__), "gen": cond.gen}}
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def loads(data: bytes):  # noqa: ANN201
    """Deserialize ``.voice`` bytes → ``(Conditionals, VoiceMeta)``."""
    import torch  # noqa: PLC0415

    try:
        blob = torch.load(io.BytesIO(data), weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Could not read .voice artifact: {exc}") from exc

    if not isinstance(blob, dict) or "meta" not in blob or "cond" not in blob:
        raise ArtifactError("Malformed .voice artifact: missing 'meta'/'cond'.")

    meta = VoiceMeta.from_dict(blob["meta"])
    Conditionals, T3Cond = _cond_classes(meta.engine)
    cond_payload = blob["cond"]
    cond = Conditionals(T3Cond(**cond_payload["t3"]), cond_payload["gen"])
    return cond, meta


def save(cond, meta: VoiceMeta, path: str | Path) -> Path:  # noqa: ANN001
    """Write an already-built ``Conditionals`` to a ``.voice`` file."""
    path = Path(path)
    path.write_bytes(dumps(cond, meta))
    return path


def load(path: str | Path):  # noqa: ANN201
    """Read a ``.voice`` file → ``(Conditionals, VoiceMeta)``."""
    return loads(Path(path).read_bytes())


def read_meta(path: str | Path) -> VoiceMeta:
    """Read just the :class:`VoiceMeta` (needs torch, not chatterbox)."""
    import torch  # noqa: PLC0415

    try:
        blob = torch.load(io.BytesIO(Path(path).read_bytes()), weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Could not read .voice artifact: {exc}") from exc
    if not isinstance(blob, dict) or "meta" not in blob:
        raise ArtifactError("Malformed .voice artifact: missing 'meta'.")
    return VoiceMeta.from_dict(blob["meta"])


def forge(model, clip_path: str | Path, meta: VoiceMeta) -> bytes:  # noqa: ANN001
    """Compute conditioning from *clip_path* on *model* and return ``.voice`` bytes.

    Turbo requires the reference clip to be **> 5 seconds** (it asserts internally).
    """
    validate_engine(meta.engine)
    model.prepare_conditionals(str(clip_path), exaggeration=meta.exaggeration)
    if getattr(model, "conds", None) is None:
        raise ArtifactError("prepare_conditionals did not produce conditioning.")
    return dumps(model.conds, meta)


def apply(model, cond) -> None:  # noqa: ANN001
    """Install a preloaded ``Conditionals`` onto *model* (voice hot-swap, no clip)."""
    device = getattr(model, "device", None)
    model.conds = cond.to(device) if device is not None else cond


def ensure_compatible(meta: VoiceMeta, engine: str) -> None:
    """Raise :class:`ArtifactError` if a *meta* voice can't run on *engine*."""
    validate_engine(engine)
    if meta.engine != engine:
        raise ArtifactError(
            f"Voice was forged for {meta.engine!r} but requested engine is {engine!r}; "
            "base and turbo artifacts are not interchangeable."
        )
