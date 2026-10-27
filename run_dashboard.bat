@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project's Python 3.12 environment...
    py -3.12 -m venv .venv
    if errorlevel 1 goto python_error
)

if not exist ".venv\.smart_quant_deps_installed" (
    echo Installing Smart-Quant dependencies. This is only needed on first run...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto install_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto install_error
    type nul > ".venv\.smart_quant_deps_installed"
)

echo Starting Smart-Quant. Keep this window open while using the dashboard.
".venv\Scripts\python.exe" -m streamlit run main.py
goto end

:python_error
echo Could not create the Python 3.12 environment. Install Python 3.12 and enable the Python Launcher, then try again.
pause
goto end

:install_error
echo Dependency installation failed. Check your internet connection and the error above, then run this file again.
pause

:end
endlocal
