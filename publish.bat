@echo off
REM Build and upload to PyPI - Windows
REM Usage: publish.bat [conda_env_name]

setlocal

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=awc

echo Activating conda environment: %ENV_NAME%
call conda activate %ENV_NAME%

echo Cleaning dist folder...
if exist dist rmdir /s /q dist

echo Building package...
python -m build
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Uploading to PyPI...
python -m twine upload dist/*

endlocal
