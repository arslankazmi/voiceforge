# VoiceForge drag-and-drop launchers

Drop an audio clip onto an icon → get a `.voice` clone next to it. These are **thin
launchers**, not standalone binaries: they call the installed `voiceforge` CLI, so the
heavy Chatterbox engine is installed **once** (shared) instead of bundled per-OS.

## Prerequisite (all platforms)

`voiceforge` must be runnable. Either:

- **Install it on PATH** (recommended for distribution):
  ```
  pipx install 'voiceforge[clone]'        # or: uv tool install 'voiceforge[clone]'
  ```
- **Or point the launcher at an existing install** via the `VOICEFORGE_BIN` env var
  (e.g. a project venv): `VOICEFORGE_BIN=/path/to/.venv/bin/voiceforge`.

First run downloads the model weights (~1.5 GB) from HuggingFace and caches them.
Turbo (default) needs a reference clip **longer than 5 seconds**.

---

## 🍎 macOS — `VoiceForge.app`

```
launchers/macos/build.sh        # builds VoiceForge.app, baking in your voiceforge path
```
Then **drag an audio file onto `VoiceForge.app`** (drop it in `/Applications` or the Dock).
It forges `yourclip.voice` next to the original and shows a result dialog. Built with
`osacompile` — no third-party tools.

## 🪟 Windows — `Forge Voice.bat`

**Drag one or more audio files onto `Forge Voice.bat`.** It forges a `.voice` next to each
and keeps the console open so you can read the result. Set `VOICEFORGE_BIN` if `voiceforge`
isn't on PATH.

## 🐧 Linux — `voiceforge-forge` + `.desktop`

1. Make the wrapper executable and (optionally) put it on PATH:
   ```
   chmod +x launchers/linux/voiceforge-forge
   ```
2. Edit `voiceforge-forge.desktop` so `Exec=` is the **absolute path** to the wrapper
   (or keep `voiceforge-forge` if it's on PATH), then drop it in `~/.local/share/applications/`
   or your desktop.
3. **Drag audio files onto the launcher.** It opens a terminal, forges each `.voice`, and
   waits so you can read the output.

---

## Why not a single compiled `.exe` per OS?

You *can* — PyInstaller/py2app can freeze `voiceforge` + torch into a standalone binary
that needs no Python. But it's **~1–2 GB per platform**, slow to build, and still downloads
model weights on first run. These launchers give the same drag-and-drop UX at a few KB by
reusing one shared engine install. If you specifically need a no-Python-required binary
(e.g. shipping to non-technical users), that PyInstaller target can be added separately.
