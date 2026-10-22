@echo off
REM One-click setup + training for the MuJoCo GET-UP task on Windows (lying down -> crouch).
REM  1. pulls mujoco_sim/ from the mujoco-walking branch into this checkout (working tree only)
REM  2. creates a conda env "premaid_mujoco" the first time
REM  3. trains, resuming from the newest checkpoint if there is one
REM Extra arguments are passed to train_mujoco.py (later ones win), e.g. --viewer --num_envs 64
setlocal
cd /d "%~dp0\.."

echo === [1/3] Fetching mujoco_sim from origin/mujoco-walking ===
git fetch origin mujoco-walking || goto :fail
git restore --source origin/mujoco-walking --worktree -- mujoco_sim ":(exclude)mujoco_sim/*.bat" || goto :fail

echo === [2/3] Python environment ===
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" || goto :fail
if not exist "%USERPROFILE%\miniconda3\envs\premaid_mujoco\python.exe" (
    call conda create -y -n premaid_mujoco -c conda-forge --override-channels python=3.11 || goto :fail
    call conda activate premaid_mujoco || goto :fail
    python -m pip install --upgrade pip
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :fail
    python -m pip install mujoco "skrl>=2.0" gymnasium imageio imageio-ffmpeg tensorboard || goto :fail
) else (
    call conda activate premaid_mujoco || goto :fail
)
python -c "import torch, mujoco, skrl; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| skrl', skrl.__version__)" || goto :fail

echo === [3/3] Training ===
cd mujoco_sim
set /a WORKERS=%NUMBER_OF_PROCESSORS%-2
if %WORKERS% LSS 1 set WORKERS=1
set /a NENVS=%WORKERS%*32
set RESUME=
if exist runs\premaid_getup_v2\*  for /r runs\premaid_getup_v2 %%f in (*.pt) do set RESUME=--resume latest
echo %NENVS% robots on %WORKERS% worker processes %RESUME%
echo TensorBoard: open another prompt, activate premaid_mujoco, run  tensorboard --logdir "%CD%\runs"
python train_mujoco.py --task getup --num_envs %NENVS% --workers %WORKERS% --timesteps 300000 %RESUME% %*
echo.
echo Training finished. Watch it with:  python play_mujoco.py --task getup
pause
exit /b 0

:fail
echo.
echo *** Setup failed - see the message above. ***
pause
exit /b 1
