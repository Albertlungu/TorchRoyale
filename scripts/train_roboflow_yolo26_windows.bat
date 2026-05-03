@echo off
setlocal

if "%ROBOFLOW_API_KEY%"=="" (
  echo ROBOFLOW_API_KEY is not set.
  echo Example:
  echo   set ROBOFLOW_API_KEY=your_key_here
  exit /b 1
)

py -3 "%~dp0train_roboflow_yolo26_windows.py" ^
  train ^
  --workspace stuff-j62hv ^
  --project clash-royale-all-enemy-cards ^
  --version 1 ^
  %*
