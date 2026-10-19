@echo off
REM Drive the newest Isaac Lab walking checkpoint with the keyboard.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" env_isaaclab
cd /d "%~dp0"
python play_walk.py %*
pause
