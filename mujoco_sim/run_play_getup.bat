@echo off
REM Watch the newest GET-UP checkpoint in the MuJoCo viewer. Backspace = new random lying pose.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" premaid_mujoco
cd /d "%~dp0"
python play_mujoco.py --task getup %*
pause
