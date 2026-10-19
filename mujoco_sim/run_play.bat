@echo off
REM Open the MuJoCo viewer with the newest checkpoint (keyboard teleop). Safe to run while training.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" premaid_mujoco
cd /d "%~dp0"
python play_mujoco.py %*
pause
