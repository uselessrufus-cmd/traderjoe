@echo off
setlocal
set BASE=%~dp0
if not exist "%BASE%config" mkdir "%BASE%config"
set /p KEY="Enter your OpenAI API key: "
if "%KEY%"=="" (
  echo No key entered.
  exit /b 1
)
> "%BASE%config\openai_key.txt" echo %KEY%
echo Saved to config\openai_key.txt
