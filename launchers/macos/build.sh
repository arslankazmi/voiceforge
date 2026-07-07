#!/usr/bin/env bash
# Build VoiceForge.app — a drag-and-drop droplet for macOS.
# Compiles the AppleScript with osacompile and bakes in the path to your installed
# `voiceforge` binary (project .venv → ~/.local/bin → PATH), so it works from Finder
# where PATH is minimal.
set -euo pipefail
cd "$(dirname "$0")"

detect_bin() {
	local repo_venv
	repo_venv="$(cd ../.. 2>/dev/null && pwd)/.venv/bin/voiceforge"
	for c in "$repo_venv" "$HOME/.local/bin/voiceforge" "$(command -v voiceforge 2>/dev/null || true)"; do
		if [ -n "$c" ] && [ -x "$c" ]; then
			echo "$c"
			return
		fi
	done
	echo "voiceforge" # last resort: hope it's on PATH at run time
}

VF="$(detect_bin)"
echo "Baking voiceforge path: $VF"

tmp="$(mktemp)"
sed "s|__VOICEFORGE_BIN__|${VF//|/\\|}|" VoiceForge-droplet.applescript >"$tmp"
rm -rf "VoiceForge.app"
osacompile -o "VoiceForge.app" "$tmp"
rm -f "$tmp"

echo "Built $(pwd)/VoiceForge.app"
echo "Drag an audio clip (>5s for Turbo) onto it to create a .voice clone next to the file."
