@echo off
REM Real-robot tools: run_deploy.bat --mode scan ^| check ^| bench ^| stand ^| run --checkpoint ...  (see deploy\README.md)
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" premaid_mujoco
python -m pip install -q pyserial
cd /d "%~dp0deploy"
python deploy_policy.py %*
pause
