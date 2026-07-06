"""Local, private library of ``.voice`` artifacts.

Layout::

    <voices_dir>/<slug>/
        voice.voice   # the portable artifact (Chatterbox conditioning + meta)
        voice.json    # torch-free sidecar: name, engine, tuning, notes, source, created

The **core is torch-free** — ``save`` / ``get`` / ``list`` / ``delete`` / ``export`` are
plain file + JSON I/O. Only :meth:`VoiceLibrary.import_artifact` (which parses an
artifact's embedded meta) needs torch. Local + private by design (lives under a
gitignored ``data/`` dir; never uploaded anywhere).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .voice_artifact import VOICE_EXT, VoiceMeta

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a human voice name into a filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "voice"


@dataclass
class StoredVoice:
    """A saved ``.voice`` artifact plus its sidecar metadata."""

    slug: str
    name: str
    engine: str
    artifact_path: str
    created: str
    meta: dict = field(default_factory=dict)
    notes: str = ""
    source: str = ""

    def settings(self) -> dict:
        """Backend ``settings`` dict that synthesizes this voice via its artifact."""
        return {
            "voice_artifact": self.artifact_path,
            "engine": self.engine,
            "temperature": float(self.meta.get("temperature", 0.8)),
            "exaggeration": float(self.meta.get("exaggeration", 0.5)),
            "cfg_weight": float(self.meta.get("cfg_weight", 0.5)),
        }


class VoiceLibrary:
    """Manages private ``.voice`` artifacts on disk under ``base_dir``."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _dir(self, slug: str) -> Path:
        return self.base_dir / slug

    def _artifact_path(self, slug: str) -> Path:
        return self._dir(slug) / f"voice{VOICE_EXT}"

    def _json_path(self, slug: str) -> Path:
        return self._dir(slug) / "voice.json"

    def _unique_slug(self, name: str) -> str:
        base = slugify(name)
        slug = base
        n = 2
        while self._dir(slug).exists():
            slug = f"{base}-{n}"
            n += 1
        return slug

    def save(
        self, meta: VoiceMeta, artifact: bytes, *, notes: str = "", source: str = ""
    ) -> StoredVoice:
        """Persist ``.voice`` *artifact* bytes + *meta* as a new library entry (torch-free)."""
        slug = self._unique_slug(meta.name)
        pack_dir = self._dir(slug)
        pack_dir.mkdir(parents=True, exist_ok=True)

        self._artifact_path(slug).write_bytes(artifact)
        stored = StoredVoice(
            slug=slug,
            name=meta.name,
            engine=meta.engine,
            artifact_path=str(self._artifact_path(slug)),
            created=datetime.now(UTC).isoformat(),
            meta=meta.to_dict(),
            notes=notes,
            source=source,
        )
        self._json_path(slug).write_text(json.dumps(asdict(stored), indent=2))
        return stored

    def import_artifact(
        self, artifact: bytes | str | Path, *, notes: str = "", source: str = ""
    ) -> StoredVoice:
        """Import a ``.voice`` blob/file — parses its embedded meta (needs torch)."""
        from .voice_artifact import loads  # noqa: PLC0415

        data = Path(artifact).read_bytes() if isinstance(artifact, (str, Path)) else artifact
        _cond, meta = loads(data)
        return self.save(meta, data, notes=notes, source=source)

    def get(self, slug: str) -> StoredVoice | None:
        meta_path = self._json_path(slug)
        if not meta_path.exists():
            return None
        return StoredVoice(**json.loads(meta_path.read_text()))

    def list(self) -> list[StoredVoice]:
        if not self.base_dir.exists():
            return []
        voices: list[StoredVoice] = []
        for child in self.base_dir.iterdir():
            if child.is_dir() and (child / "voice.json").exists():
                sv = self.get(child.name)
                if sv is not None:
                    voices.append(sv)
        voices.sort(key=lambda v: v.created, reverse=True)
        return voices

    def export(self, slug: str) -> bytes | None:
        path = self._artifact_path(slug)
        return path.read_bytes() if path.exists() else None

    def delete(self, slug: str) -> bool:
        pack_dir = self._dir(slug)
        if not pack_dir.exists():
            return False
        shutil.rmtree(pack_dir)
        return True


_library: VoiceLibrary | None = None


def get_library() -> VoiceLibrary:
    """Lazy singleton rooted at ``$VOICEFORGE_VOICES_DIR`` (default ``data/voices``)."""
    global _library
    if _library is None:
        _library = VoiceLibrary(os.environ.get("VOICEFORGE_VOICES_DIR", "data/voices"))
    return _library
