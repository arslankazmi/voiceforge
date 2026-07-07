@echo off
REM VoiceForge drag-and-drop launcher (Windows).
REM Drag one or more audio files onto this .bat icon to forge a .voice clone next
REM to each file. Requires `voiceforge` on PATH (pip/pipx install 'voiceforge[clone]')
REM or set VOICEFORGE_BIN to the full path of the voiceforge executable.
setlocal enabledelayedexpansion

set "VF=%VOICEFORGE_BIN%"
if "%VF%"=="" set "VF=voiceforge"

if "%~1"=="" (
  echo Drag an audio clip ^(WAV/MP3, longer than 5s for Turbo^) onto this icon
  echo to create a .voice clone next to it.
  echo.
  pause
  exit /b 1
)

:loop
if "%~1"=="" goto done
set "OUT=%~dpn1.voice"
echo Forging voice from "%~nx1" ...
"%VF%" forge "%~1" -o "%OUT%"
if errorlevel 1 (
  echo   ^!^! Forge failed for "%~nx1"
) else (
  echo   -^> %OUT%
)
shift
goto loop

:done
echo.
echo Done.
pause
