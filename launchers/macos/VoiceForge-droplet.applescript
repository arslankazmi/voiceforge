-- VoiceForge drag-and-drop droplet (macOS).
-- Drop audio file(s) onto this app icon to forge a .voice clone next to each file.
-- Built by build.sh, which bakes the path to your installed `voiceforge` binary.

property vfBin : "__VOICEFORGE_BIN__" -- replaced by build.sh; resolveBin() falls back

on run
	display dialog "VoiceForge" & return & return & ¬
		"Drag an audio clip onto this app icon to create a .voice clone next to it." & ¬
		return & return & "(Turbo needs a clip longer than 5 seconds.)" ¬
		buttons {"OK"} default button 1
end run

on open theItems
	set vf to resolveBin()
	repeat with anItem in theItems
		set p to POSIX path of anItem
		set outPath to (do shell script "p=" & quoted form of p & "; echo \"${p%.*}.voice\"")
		try
			display notification "Forging voice…" with title "VoiceForge"
			set result to do shell script vf & " forge " & quoted form of p & ¬
				" -o " & quoted form of outPath & " 2>&1"
			display dialog "✅ Created:" & return & outPath & return & return & result ¬
				buttons {"OK"} default button 1 with title "VoiceForge"
		on error errMsg
			display dialog "⚠️ Forge failed:" & return & errMsg ¬
				buttons {"OK"} default button 1 with title "VoiceForge" with icon caution
		end try
	end repeat
end open

on resolveBin()
	-- 1) baked-in path, 2) $VOICEFORGE_BIN, 3) common install dirs, 4) bare command
	if vfBin is not "__VOICEFORGE_BIN__" and vfBin is not "" then
		try
			do shell script "test -x " & quoted form of vfBin
			return quoted form of vfBin
		end try
	end if
	set candidates to {}
	try
		set envBin to do shell script "echo \"$VOICEFORGE_BIN\""
		if envBin is not "" then set end of candidates to envBin
	end try
	set home to do shell script "echo $HOME"
	set candidates to candidates & {home & "/.local/bin/voiceforge", ¬
		"/opt/homebrew/bin/voiceforge", "/usr/local/bin/voiceforge"}
	repeat with c in candidates
		try
			do shell script "test -x " & quoted form of c
			return quoted form of c
		end try
	end repeat
	return "voiceforge"
end resolveBin
