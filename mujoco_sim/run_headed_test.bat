@echo off
REM Short training run with a live MuJoCo window on robot #0, to check everything works.
call "%~dp0run_training_windows.bat" --viewer --num_envs 64 --timesteps 20000
