@echo off
REM Export the newest mjlab get-up checkpoint to TorchScript and watch it in the CPU MuJoCo sim (same as the real robot loop).
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" premaid_mujoco
cd /d "%~dp0"
for /f "delims=" %%f in ('dir /b /s /o:d logs\mjlab\premaid_getup\model_*.pt') do set CKPT=%%f
echo Using %CKPT%
python mjlab_premaid\export.py "%CKPT%" -o getup_policy_mjlab.pt
cd mujoco_sim
python deploy\deploy_policy.py --mode getup --getup_checkpoint ..\getup_policy_mjlab.pt --sim --sim_randomize --render %*
pause
