@echo off
REM Get-up training on the GPU with mjlab (MuJoCo Warp), inside WSL2 Ubuntu.
REM One-time: in an admin PowerShell run  wsl --install -d Ubuntu  and reboot, then open Ubuntu once to make a user.
REM Extra args are passed through, e.g.  run_mjlab_getup.bat --agent.max-iterations 5000
REM Logs + checkpoints: logs\mjlab\premaid_getup\<date>\model_*.pt   (tensorboard --logdir logs\mjlab)
cd /d "%~dp0"
for /f "delims=" %%i in ('wsl wslpath -a "%CD%"') do set REPO=%%i
wsl bash -lc "cd '%REPO%' && ./mjlab_premaid/train_getup.sh %*"
pause
