@echo off
REM Train the fixed Isaac Lab walking task. Put this next to premaidai.usd (your IsaacLab folder).
REM Extra args go to train_walk.py, e.g.  run_isaac_walk.bat --resume latest
REM To watch training live instead:  python train_walk.py --num_envs 64   (no --headless)
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" env_isaaclab
cd /d "%~dp0"
python train_walk.py --headless %*
pause
